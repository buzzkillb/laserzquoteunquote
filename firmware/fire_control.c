/*
 * firmware/fire_control.c -- laserzquoteunquote MCU fire control
 * ================================================================
 * Portable C99 core implementing the hardware/protocol.py contract:
 *
 *   -> F,<seq>,<az_md>,<el_md>,<dwell_us>,<power_cW>   fire command
 *   -> A,<seq>                                         abort
 *   -> R,<seq>                                         (re)arm
 *   -> S                                               status request
 *   -> Z,<az_md>,<el_md>,<r_md> / Z,END                veto-zone push
 *   <- T,<seq>,<az_md>,<el_md>,<flags>,<heat_cP>,<shots>,<beam_ms>
 *   <- V,<reason>                                      veto notice
 *
 * Design (from RIG.md / AUDIT.md):
 *   - Fast loop runs the veto chain INDEPENDENTLY of the Pi. The Pi
 *     proposes (az, el, dwell, power); the MCU disposes (veto authority).
 *   - Arm discipline: the core boots DISARMED. "R" arms (unless estop).
 *     Abort and watchdog DISARM. Firing never arms. This means a crashed
 *     Pi can at most leave the beam off.
 *   - Watchdog: no valid frame for 500 ms while armed -> disarm.
 *   - Thermal: heat rises with power*dwell (same units as the Python
 *     sim's model: dwell_s * 4 * power_w/2, in centi-percent), decays
 *     5500 cP/s, refuses new shots above 9200 cP (92%).
 *   - Abort/estop truncate the shot; delivered dwell is what counts for
 *     beam-time and heat accounting.
 *
 * HAL-free: calls only the ops in fire_control.h so it unit-tests on a
 * host (clang/gcc) and compiles unchanged inside an ESP32 main.c.
 */

#include "fire_control.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* ------------------------------ constants ------------------------------ */
#define WATCHDOG_MS 500
#define HEAT_LIMIT 9200         /* refuse new shots above 92% */
#define MAX_ZONES 8
#define FLAG_ARMED 1
#define FLAG_FIRING 2
#define FLAG_ESTOP 4
#define FLAG_THERMAL 8

/* ------------------------------ state ------------------------------ */
typedef struct {
    int armed;
    int firing;
    int estop;
    uint16_t last_seq;
    uint32_t ms;              /* monotonic ms (tick count) */
    uint32_t last_rx_ms;      /* ms of last valid frame (watchdog food) */
    int32_t az_md, el_md;     /* commanded galvo angles */
    int32_t dwell_us;         /* dwell for shot in progress */
    int32_t dwell_done_us;    /* delivered so far */
    int32_t power_cw;         /* commanded beam power */
    int32_t heat_hcp;         /* half-centi-percent accumulator: decay is
                               * 5.5 cP/tick; stored in 0.5 cP units so
                               * every tick decays exactly 11 hcp with
                               * zero drift vs sim3d.py's 5500 cP/s */
    uint32_t shot_count;
    uint32_t beam_ms_total;
    char veto_reason[24];
} fc_state;

static fc_state S;
static fire_hal H;

typedef struct { int32_t az_md, el_md, r_md; } fc_zone;
static fc_zone ZN[MAX_ZONES];      /* active zones (pushed while disarmed) */
static int zn_n;
static fc_zone ZS[MAX_ZONES];      /* scratch, committed on Z,END */
static int zs_n;

/* ------------------------------ helpers ------------------------------ */
static int32_t clamp_i32(int32_t v, int32_t lo, int32_t hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

void fc_reset(void) {
    memset(&S, 0, sizeof S);
    zn_n = 0; zs_n = 0;
}

int fc_init(const fire_hal *hal) {
    if (!hal || !hal->laser_on || !hal->laser_off ||
        !hal->galvo_to || !hal->ms || hal->beam_cw_limit <= 0) return -1;
    H = *hal;
    fc_reset();
    return 0;
}

static void beam_off(void) {
    if (S.firing) {
        H.laser_off();
        S.firing = 0;
    }
}

/* End a shot (completed or truncated): account beam time + heat for the
 * dwell actually delivered. */
static void end_shot(void) {
    beam_off();
    S.shot_count++;
    S.beam_ms_total += (uint32_t)(S.dwell_done_us / 1000);
    /* heat += power_cw * dwell_us / 5000  ==  dwell_s * 4 * power_w/2
     * expressed in centi-percent (matches sim3d.py's thermal model) */
    S.heat_hcp += (int32_t)(((int64_t)S.power_cw * S.dwell_done_us) / 2500);
    if (S.heat_hcp > 20000) S.heat_hcp = 20000;
    S.dwell_done_us = 0;
}

/* ------------------------------ veto chain ------------------------------ */
static int zone_hit(int32_t az_md, int32_t el_md) {
    for (int i = 0; i < zn_n; i++) {
        int32_t daz = az_md - ZN[i].az_md;
        if (daz > 180000) daz -= 360000;
        if (daz < -180000) daz += 360000;
        int32_t de = el_md - ZN[i].el_md;
        int64_t rr = (int64_t)daz * daz + (int64_t)de * de;
        if (rr <= (int64_t)ZN[i].r_md * ZN[i].r_md) return 1;
    }
    return 0;
}

static const char *veto_check(int32_t power_cw, int32_t az_md, int32_t el_md) {
    if (S.estop) return "estop";
    if (!S.armed) return "disarmed";
    if (zone_hit(az_md, el_md)) return "human_in_beam";
    if (S.heat_hcp >= 2 * HEAT_LIMIT) return "thermal";
    if (power_cw > H.beam_cw_limit) return "power";
    return NULL;
}

/* ------------------------------ frame parsing ------------------------------ */
int fc_on_line(const char *line, fc_tx_fn out, void *ud) {
    if (!line || !out) return 0;

    /* A,<seq> -- abort: always honored, even mid-dwell; disarms */
    if (line[0] == 'A' && line[1] == ',') {
        if (S.firing) end_shot();          /* truncate + account delivery */
        S.armed = 0;
        S.last_seq = (uint16_t)strtoul(line + 2, NULL, 10);
        S.last_rx_ms = S.ms;
        return 0;
    }

    /* R,<seq> -- (re)arm */
    if (line[0] == 'R' && line[1] == ',') {
        S.last_seq = (uint16_t)strtoul(line + 2, NULL, 10);
        S.last_rx_ms = S.ms;
        if (!S.estop) S.armed = 1;
        return 0;
    }

    /* Z,... -- veto-zone push, mirroring SimInterlock semantics:
     *   Z,<az_md>,<el_md>,<radius_md>   one cone;  Z,END commits.
     * Zones are immutable while armed; a commit while armed keeps the
     * previous set. Accumulation is scratch-only until Z,END. */
    if (line[0] == 'Z' && line[1] == ',') {
        S.last_rx_ms = S.ms;
        const char *rest = line + 2;
        if (rest[0] == 'E' && rest[1] == 'N' && rest[2] == 'D') {
            if (!S.armed) {
                zn_n = zs_n;
                for (int i = 0; i < zs_n; i++) ZN[i] = ZS[i];
            }
            zs_n = 0;
            return 0;
        }
        if (zs_n < MAX_ZONES) {
            char *p;
            long a = strtol(rest, &p, 10);
            long e = (p && *p == ',') ? strtol(p + 1, &p, 10) : 0;
            long r = (p && *p == ',') ? strtol(p + 1, NULL, 10) : 0;
            ZS[zs_n].az_md = (int32_t)a;
            ZS[zs_n].el_md = (int32_t)e;
            ZS[zs_n].r_md = (int32_t)r;
            zs_n++;
        }
        return 0;
    }

    /* S -- status request (tolerate CRLF: real UARTs terminate lines) */
    if (line[0] == 'S' && (line[1] == 0 || line[1] == '\r' || line[1] == '\n')) {
        char t[80];
        int flags = (S.armed ? FLAG_ARMED : 0) |
                    (S.firing ? FLAG_FIRING : 0) |
                    (S.estop ? FLAG_ESTOP : 0) |
                    (S.heat_hcp >= 2 * HEAT_LIMIT ? FLAG_THERMAL : 0);
        int n = snprintf(t, sizeof t, "T,%u,%ld,%ld,%d,%ld,%lu,%lu",
                         (unsigned)S.last_seq,
                         (long)S.az_md, (long)S.el_md, flags,
                         (long)(S.heat_hcp / 2),
                         (unsigned long)S.shot_count,
                         (unsigned long)S.beam_ms_total);
        S.last_rx_ms = S.ms;
        out(t, n, ud);
        return 1;
    }

    /* F,<seq>,<az_md>,<el_md>,<dwell_us>,<power_cW> */
    if (line[0] == 'F' && line[1] == ',') {
        /* require exactly 5 comma-separated fields (plus the F) */
        int commas = 0;
        for (const char *q = line; *q; q++) commas += (*q == ',');
        if (commas != 5) return 0;
        char *p;
        long seq = strtol(line + 2, &p, 10);
        long az_md = (p && *p == ',') ? strtol(p + 1, &p, 10) : 0;
        long el_md = (p && *p == ',') ? strtol(p + 1, &p, 10) : 0;
        long dwell_us = (p && *p == ',') ? strtol(p + 1, &p, 10) : 0;
        long power_cw = (p && *p == ',') ? strtol(p + 1, NULL, 10) : 0;

        S.last_seq = (uint16_t)seq;
        S.last_rx_ms = S.ms;

        /* A shot already in flight is closed out BEFORE this command is
         * judged. Its delivered dwell has to be charged to heat and
         * beam-time, or a Pi that re-commands faster than its own dwell
         * keeps the beam lit while heat reads zero and the thermal veto
         * never trips. Ending it here also guarantees the beam is off
         * while the galvo slews to the new target. */
        if (S.firing) end_shot();

        const char *veto = veto_check((int32_t)power_cw, (int32_t)az_md,
                                      (int32_t)el_md);
        if (veto) {
            snprintf(S.veto_reason, sizeof S.veto_reason, "%s", veto);
            char v[24];
            int n = snprintf(v, sizeof v, "V,%s", veto);
            out(v, n, ud);
            return 1;
        }

        /* program the shot: galvo moves FIRST, then beam on */
        S.az_md = clamp_i32((int32_t)az_md, -180000, 180000);
        S.el_md = clamp_i32((int32_t)el_md, -90000, 90000);
        S.dwell_us = clamp_i32((int32_t)dwell_us, 0, 1000000);
        S.dwell_done_us = 0;
        S.power_cw = clamp_i32((int32_t)power_cw, 0, H.beam_cw_limit);
        H.galvo_to(S.az_md, S.el_md);
        S.firing = 1;
        H.laser_on(S.power_cw);
        return 0;
    }

    return 0;   /* garbage lines are ignored (never crash on the wire) */
}

/* ------------------------------ tick: the fast loop ------------------------------ */
void fc_tick_1khz(void) {
    S.ms++;
    if (S.firing) {
        S.dwell_done_us += 1000;               /* us delivered this tick */
        if (S.dwell_done_us >= S.dwell_us) end_shot();
    }

    /* thermal decay, every tick: exactly 11 hcp = 5.5 cP (5500 cP/s),
     * zero integer drift vs sim3d.py's 0.55/s model */
    S.heat_hcp -= 11;
    if (S.heat_hcp < 0) S.heat_hcp = 0;

    /* watchdog: armed with no valid frame for 500 ms -> safe state */
    if (S.armed && (uint32_t)(S.ms - S.last_rx_ms) > WATCHDOG_MS) {
        if (S.firing) end_shot();
        S.armed = 0;
        snprintf(S.veto_reason, sizeof S.veto_reason, "watchdog");
    }
}

/* ------------------------------ estop + accessors ------------------------------ */
void fc_estop(void) {
    if (S.firing) end_shot();
    S.estop = 1;
    S.armed = 0;
}

void fc_clear_estop(void) { S.estop = 0; }

void fc_note_rx(void) { S.last_rx_ms = S.ms; }

const char *fc_veto_reason(void) { return S.veto_reason; }
int fc_armed(void) { return S.armed; }
int fc_firing(void) { return S.firing; }
int32_t fc_heat(void) { return S.heat_hcp / 2; }
uint32_t fc_shots(void) { return S.shot_count; }
uint32_t fc_beam_ms(void) { return S.beam_ms_total; }
int fc_estop_active(void) { return S.estop; }
