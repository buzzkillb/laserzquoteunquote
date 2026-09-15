/*
 * test_fire_control.c -- host-side test bench for the firmware core
 * =================================================================
 * Exercises the exact wire format hardware/protocol.py emits, plus the
 * safety state machine: arm discipline, veto chain, thermal saturation,
 * watchdog, estop, abort truncation, and heat accounting vs the Python
 * sim's thermal model.
 *
 * Build & run:
 *   cc -Wall -Wextra -O2 -o test_fw test_fire_control.c fire_control.c && ./test_fw
 */
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "fire_control.h"

static int laser_state;
static int32_t last_power;
static int galvo_moves;

static void hal_on(int32_t pw) { laser_state = 1; last_power = pw; }
static void hal_off(void) { laser_state = 0; }
static void hal_galvo(int32_t az, int32_t el) { (void)az; (void)el; galvo_moves++; }
static uint32_t hal_ms(void) { return 0; }   /* watchdog driven via ticks */

static char txbuf[256];
static void capture(const char *b, int n, void *ud) {
    (void)ud;
    memcpy(txbuf, b, n); txbuf[n] = 0;
}

static int checks = 0;
#define CHECK(cond) do { \
    if (!(cond)) { printf("FAIL line %d: %s\n", __LINE__, #cond); return 1; } \
    checks++; \
} while (0)

/* run the fast loop until the current shot ends (or timeout), feeding
 * the watchdog with status polls every 100 ms -- exactly what the real
 * Pi slow loop does (60 Hz status requests). NOTE: a >500 ms shot with
 * NO polls gets cut by the watchdog; that behavior is tested at (7). */
static void run_shot_out(void) {
    for (int i = 0; i < 1000005 && fc_firing(); i++) {
        fc_tick_1khz();
        if (i % 100 == 0) fc_on_line("S", capture, 0);
    }
}

int main(void) {
    fire_hal h = { hal_on, hal_off, hal_galvo, hal_ms, 200 };
    assert(fc_init(&h) == 0);

    /* ---- 0. boots disarmed: F before R is vetoed ---- */
    CHECK(fc_armed() == 0);
    CHECK(fc_on_line("F,1,0,0,1000,200", capture, 0) == 1);
    CHECK(strncmp(txbuf, "V,disarmed", 10) == 0);
    CHECK(laser_state == 0);

    /* ---- 1. arm, then fire: exact protocol.py frame ---- */
    fc_on_line("R,2", capture, 0);
    CHECK(fc_armed() == 1);
    /* 30 deg az, 5 deg el, 30 ms dwell, 2.00 W */
    CHECK(fc_on_line("F,42,30000,5000,30000,200", capture, 0) == 0);
    CHECK(fc_firing() == 1);
    CHECK(laser_state == 1 && last_power == 200);
    CHECK(galvo_moves == 1);

    /* ---- 2. dwell completes: shot counted, heat matches sim model ---- */
    for (int i = 0; i < 31; i++) fc_tick_1khz();
    CHECK(fc_firing() == 0);
    CHECK(fc_shots() == 1);
    CHECK(fc_beam_ms() == 30);
    /* heat = 200*30000/5000 = 1200 cP at end of shot, minus ~3 ticks of
     * 5.5 cP decay; assert the sim-matching band, not tick parity */
    CHECK(fc_heat() > 1100 && fc_heat() <= 1200);

    /* ---- 3. status frame: T,<seq>,<az>,<el>,<flags>,<heat>,<shots>,<beam> ---- */
    fc_on_line("S", capture, 0);
    CHECK(strncmp(txbuf, "T,42,30000,5000,1,", 18) == 0);
    CHECK(strstr(txbuf, ",1,30") != NULL);          /* shots=1, beam=30 */

    /* ---- 4. veto: power over the 2 W diode limit ---- */
    CHECK(fc_on_line("F,43,0,0,1000,500", capture, 0) == 1);
    CHECK(strncmp(txbuf, "V,power", 7) == 0);
    CHECK(fc_firing() == 0);            /* nothing fired */
    CHECK(fc_armed() == 1);             /* still armed */

    /* ---- 5. abort mid-dwell: beam cut NOW, delivery accounted ---- */
    CHECK(fc_on_line("F,44,0,0,100000,200", capture, 0) == 0);
    for (int i = 0; i < 50; i++) fc_tick_1khz();    /* 50 ms in */
    CHECK(fc_firing() == 1);
    CHECK(fc_on_line("A,44", capture, 0) == 0);
    CHECK(laser_state == 0);            /* beam CUT immediately */
    CHECK(fc_armed() == 0);             /* abort disarms */
    CHECK(fc_beam_ms() == 80);          /* 30 delivered + 50 delivered */
    CHECK(fc_shots() == 2);             /* truncated shot still counted */

    /* ---- 6. estop: latching, blocks even after rearm attempt ---- */
    fc_on_line("R,45", capture, 0);
    CHECK(fc_armed() == 1);
    fc_estop();
    CHECK(laser_state == 0 && fc_estop_active() == 1);
    CHECK(fc_on_line("F,46,0,0,1000,200", capture, 0) == 1);
    CHECK(strncmp(txbuf, "V,estop", 7) == 0);
    fc_on_line("R,47", capture, 0);
    CHECK(fc_armed() == 0);             /* R cannot override estop */
    fc_clear_estop();
    fc_on_line("R,48", capture, 0);
    CHECK(fc_on_line("F,49,0,0,1000,200", capture, 0) == 0);
    CHECK(fc_firing() == 1);
    run_shot_out();
    CHECK(fc_firing() == 0);

    /* ---- 7. watchdog: armed + 500 ms silence -> beam off, disarmed ---- */
    fc_on_line("R,50", capture, 0);
    CHECK(fc_on_line("F,51,0,0,100000,200", capture, 0) == 0);
    CHECK(fc_firing() == 1);
    for (int i = 0; i < 520; i++) fc_tick_1khz();   /* no frames: watchdog */
    CHECK(fc_firing() == 0);
    CHECK(fc_armed() == 0);
    CHECK(strncmp(fc_veto_reason(), "watchdog", 8) == 0);
    CHECK(laser_state == 0);

    /* ---- 8. thermal: 1 s full-power shot saturates, next is vetoed,
     *         decay re-enables firing ---- */
    fc_on_line("R,60", capture, 0);
    CHECK(fc_on_line("F,61,0,0,1000000,200", capture, 0) == 0);
    run_shot_out();
    CHECK(fc_heat() >= 9900 && fc_heat() <= 10000);  /* saturated, never over */
    CHECK(fc_on_line("F,62,0,0,1000,200", capture, 0) == 1);
    CHECK(strncmp(txbuf, "V,thermal", 9) == 0);
    for (int i = 0; i < 2000; i++) fc_tick_1khz();   /* decay ~11000 cP */
    CHECK(fc_heat() == 0);
    CHECK(fc_armed() == 0);   /* silence during decay -> watchdog disarmed */
    fc_on_line("R,63", capture, 0);                  /* Pi re-arms */
    CHECK(fc_on_line("F,63,0,0,1000,200", capture, 0) == 0);  /* fires again */
    run_shot_out();

    /* ---- 9. garbage lines: ignored, never crash, never fire ---- */
    CHECK(fc_on_line("", capture, 0) == 0);
    CHECK(fc_on_line("X,1,2", capture, 0) == 0);
    CHECK(fc_on_line("F,garbage", capture, 0) == 0);   /* malformed F */
    CHECK(fc_on_line("F,1,2,3", capture, 0) == 0);     /* too few fields */
    CHECK(fc_on_line(NULL, capture, 0) == 0);
    CHECK(fc_firing() == 0);

    printf("ALL %d CHECKS PASSED\n", checks);
    return 0;
}
