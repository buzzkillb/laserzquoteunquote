/* fire_control.h -- public contract of the MCU fire-control core */
#ifndef FIRE_CONTROL_H
#define FIRE_CONTROL_H

#include <stdint.h>

/* HAL: the only hardware access the core is allowed. A board port fills
 * these in main.c; the host test suite fills them with fakes. */
typedef struct {
    void (*laser_on)(int32_t power_cw);
    void (*laser_off)(void);
    void (*galvo_to)(int32_t az_md, int32_t el_md);
    uint32_t (*ms)(void);          /* monotonic ms */
    int32_t beam_cw_limit;         /* 200 = 2 W diode, 500 = Pro */
} fire_hal;

/* line handler: called with complete ASCII lines from the UART;
 * implementations should snprintf a reply and out() it. */
typedef void (*fc_tx_fn)(const char *buf, int len, void *ud);

int  fc_init(const fire_hal *hal);
void fc_reset(void);
int  fc_on_line(const char *line, fc_tx_fn out, void *ud);
void fc_tick_1khz(void);
void fc_estop(void);
void fc_clear_estop(void);
void fc_note_rx(void);            /* call on ANY valid frame from the Pi */

const char *fc_veto_reason(void);
int  fc_armed(void);
int  fc_firing(void);
int32_t fc_heat(void);
uint32_t fc_shots(void);
uint32_t fc_beam_ms(void);
int  fc_estop_active(void);

#endif
