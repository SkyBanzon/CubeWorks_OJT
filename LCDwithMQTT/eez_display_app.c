#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_log.h"
#include "esp_random.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_eth.h"
#include "esp_eth_mac.h"
#include "esp_eth_phy.h"
#include "nvs_flash.h"
#include "mqtt_client.h"
#include "driver/gpio.h"
#include "lvgl.h"
#include "bsp/esp-bsp.h"

static const char *TAG = "lcd_mqtt_eth";

/*
 * VERSION 5 — Native Ethernet MQTT (ESP32-P4-ETH board)
 *
 * The ESP32-P4-ETH has a built-in RMII Ethernet MAC connected to an
 * external PHY (typically IP101 or LAN8720). This firmware uses that
 * interface directly — no WiFi needed.
 *
 * Network path:
 *   ESP32-P4-ETH ──(RJ45)──► Ethernet laptop (ICS/DHCP)
 *                             └──(WiFi/LAN)──► Type-C laptop (Mosquitto)
 *
 * Topics published:
 *   esp32/color   → "red", "green", "blue", etc.
 *   esp32/random  → 8-character alphanumeric string
 *
 * Setup:
 *   1. Set MQTT_BROKER_URI to the Type-C laptop's IP.
 *      Run ipconfig on the Type-C laptop and use the IP on the adapter
 *      that faces the shared network (likely 192.168.137.x if ICS is on).
 *   2. Mosquitto must listen on 0.0.0.0 (not 127.0.0.1):
 *         listener 1883
 *         allow_anonymous true
 *   3. In sdkconfig / menuconfig enable:
 *         Component config → Ethernet → Support ESP32 internal EMAC controller
 *         Component config → Ethernet → PHY → (your PHY chip)
 *
 * menuconfig requirements (same as before):
 *   Component config → Hardware Settings → Minimum Supported ESP32-P4 Rev = v1.0
 *   Component config → Board Support Package(ESP32-P4) → Waveshare 10.1-DSI-TOUCH-A
 *   Component config → ESP PSRAM → Make RAM allocatable using malloc()
 *   Serial flasher config → Flash size = 32 MB
 */

/* ── User config ─────────────────────────────────────────────────────────── */
#define MQTT_BROKER_URI  "mqtt://10.40.71.108:1883"  /* ← Type-C laptop IP */

#define MQTT_TOPIC_COLOR  "esp32/color"
#define MQTT_TOPIC_RANDOM "esp32/random"

/*
 * PHY config — ESP32-P4-ETH typically uses IP101GRI.
 * If your board uses LAN8720 instead, change ETH_PHY_IP101 → ETH_PHY_LAN8720
 * and adjust PHY_ADDR if needed (check your board schematic).
 */
#define ETH_PHY_TYPE      ETH_PHY_IP101
#define ETH_PHY_ADDR      1
#define ETH_PHY_MDC_GPIO  31
#define ETH_PHY_MDIO_GPIO 52
#define ETH_PHY_RST_GPIO  51   /* -1 if no reset pin wired */
/* ETH_CLK_MODE removed — ESP-IDF v5.5 uses emac_config.interface instead */
/* ─────────────────────────────────────────────────────────────────────────── */

/* Event group bits */
#define ETH_CONNECTED_BIT    BIT0
#define ETH_DISCONNECTED_BIT BIT1

static EventGroupHandle_t           s_eth_event_group;
static esp_mqtt_client_handle_t     s_mqtt_client = NULL;
static bool                         s_mqtt_ready  = false;

/* UI handles */
static lv_obj_t *status_label    = NULL;
static lv_obj_t *last_sent_label = NULL;

/* ── Button definitions ───────────────────────────────────────────────────── */
typedef struct {
    const char *name;
    const char *payload;
    uint32_t    color_hex;
    uint32_t    pressed_hex;
} color_button_t;

static const color_button_t color_buttons[] = {
    {"RED",    "red",    0xE74C3C, 0x922B21},
    {"GREEN",  "green",  0x27AE60, 0x196F3D},
    {"BLUE",   "blue",   0x2980B9, 0x1B4F72},
    {"YELLOW", "yellow", 0xF1C40F, 0x9A7D0A},
    {"PURPLE", "purple", 0x8E44AD, 0x5B2C6F},
    {"ORANGE", "orange", 0xE67E22, 0xA04000},
};

/* ── UI helpers ──────────────────────────────────────────────────────────── */
static void set_status(const char *text)
{
    if (status_label) {
        lv_label_set_text(status_label, text);
    }
}

static void set_last_sent(const char *prefix, const char *value)
{
    static char buf[96];
    snprintf(buf, sizeof(buf), "%s%s", prefix, value);
    if (last_sent_label) {
        lv_label_set_text(last_sent_label, buf);
    }
}

/* ── MQTT publish helpers ────────────────────────────────────────────────── */
static void mqtt_publish_color(const char *color)
{
    if (!s_mqtt_ready || s_mqtt_client == NULL) {
        ESP_LOGW(TAG, "MQTT not ready — color not sent");
        set_status("MQTT not connected!");
        return;
    }
    int msg_id = esp_mqtt_client_publish(
        s_mqtt_client, MQTT_TOPIC_COLOR, color, 0, 1, 0);
    if (msg_id >= 0) {
        ESP_LOGI(TAG, "Published color: %s (id=%d)", color, msg_id);
    } else {
        ESP_LOGE(TAG, "Publish failed for color: %s", color);
        set_status("Publish failed!");
    }
}

static void mqtt_publish_random(const char *text)
{
    if (!s_mqtt_ready || s_mqtt_client == NULL) {
        ESP_LOGW(TAG, "MQTT not ready — random not sent");
        set_status("MQTT not connected!");
        return;
    }
    int msg_id = esp_mqtt_client_publish(
        s_mqtt_client, MQTT_TOPIC_RANDOM, text, 0, 1, 0);
    if (msg_id >= 0) {
        ESP_LOGI(TAG, "Published random: %s (id=%d)", text, msg_id);
    } else {
        ESP_LOGE(TAG, "Publish failed for random: %s", text);
        set_status("Publish failed!");
    }
}

/* ── Random string ───────────────────────────────────────────────────────── */
static void generate_random_string(char *out, size_t out_len)
{
    const char charset[]     = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    const size_t charset_len = sizeof(charset) - 1;
    if (out_len == 0) return;
    for (size_t i = 0; i < out_len - 1; i++) {
        out[i] = charset[esp_random() % charset_len];
    }
    out[out_len - 1] = '\0';
}

/* ── Button callbacks ────────────────────────────────────────────────────── */
static void color_btn_event_cb(lv_event_t *e)
{
    if (lv_event_get_code(e) != LV_EVENT_PRESSED) return;
    const color_button_t *cfg = (const color_button_t *)lv_event_get_user_data(e);
    if (cfg == NULL) return;
    mqtt_publish_color(cfg->payload);
    static char status_buf[64];
    snprintf(status_buf, sizeof(status_buf), "Pressed: %s", cfg->name);
    set_status(status_buf);
    set_last_sent("Last sent color: ", cfg->payload);
}

static void random_btn_event_cb(lv_event_t *e)
{
    if (lv_event_get_code(e) != LV_EVENT_PRESSED) return;
    char random_text[9];
    generate_random_string(random_text, sizeof(random_text));
    mqtt_publish_random(random_text);
    set_status("Pressed: RANDOM");
    set_last_sent("Last random: ", random_text);
}

/* ── MQTT event handler ──────────────────────────────────────────────────── */
static void mqtt_event_handler(void *handler_args,
                               esp_event_base_t base,
                               int32_t event_id,
                               void *event_data)
{
    esp_mqtt_event_handle_t event = (esp_mqtt_event_handle_t)event_data;
    switch ((esp_mqtt_event_id_t)event_id) {

    case MQTT_EVENT_CONNECTED:
        ESP_LOGI(TAG, "MQTT connected");
        s_mqtt_ready = true;
        bsp_display_lock(0);
        set_status("Status: MQTT Connected");
        bsp_display_unlock();
        break;

    case MQTT_EVENT_DISCONNECTED:
        ESP_LOGW(TAG, "MQTT disconnected");
        s_mqtt_ready = false;
        bsp_display_lock(0);
        set_status("Status: MQTT Disconnected");
        bsp_display_unlock();
        break;

    case MQTT_EVENT_PUBLISHED:
        ESP_LOGI(TAG, "MQTT published, msg_id=%d", event->msg_id);
        break;

    case MQTT_EVENT_ERROR:
        ESP_LOGE(TAG, "MQTT error");
        break;

    default:
        break;
    }
}

/* ── Ethernet + IP event handler ─────────────────────────────────────────── */
static void eth_event_handler(void *arg,
                              esp_event_base_t event_base,
                              int32_t event_id,
                              void *event_data)
{
    if (event_base == ETH_EVENT) {
        switch (event_id) {
        case ETHERNET_EVENT_CONNECTED:
            ESP_LOGI(TAG, "Ethernet link up — waiting for DHCP...");
            bsp_display_lock(0);
            set_status("Status: Ethernet up, getting IP...");
            bsp_display_unlock();
            break;

        case ETHERNET_EVENT_DISCONNECTED:
            ESP_LOGW(TAG, "Ethernet link down");
            s_mqtt_ready = false;
            xEventGroupSetBits(s_eth_event_group, ETH_DISCONNECTED_BIT);
            bsp_display_lock(0);
            set_status("Status: Ethernet disconnected");
            bsp_display_unlock();
            break;

        case ETHERNET_EVENT_START:
            ESP_LOGI(TAG, "Ethernet started");
            break;

        case ETHERNET_EVENT_STOP:
            ESP_LOGI(TAG, "Ethernet stopped");
            break;

        default:
            break;
        }

    } else if (event_base == IP_EVENT && event_id == IP_EVENT_ETH_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
        xEventGroupSetBits(s_eth_event_group, ETH_CONNECTED_BIT);

        static char ip_buf[48];
        snprintf(ip_buf, sizeof(ip_buf),
                 "ETH OK — IP: " IPSTR, IP2STR(&event->ip_info.ip));
        bsp_display_lock(0);
        set_status(ip_buf);
        bsp_display_unlock();
    }
}

/* ── Ethernet init ───────────────────────────────────────────────────────── */
static bool eth_init(void)
{
    s_eth_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    /* Create default Ethernet netif */
    esp_netif_config_t netif_cfg = ESP_NETIF_DEFAULT_ETH();
    esp_netif_t *eth_netif = esp_netif_new(&netif_cfg);

    /* Register event handlers */
    ESP_ERROR_CHECK(esp_event_handler_register(
        ETH_EVENT, ESP_EVENT_ANY_ID, &eth_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(
        IP_EVENT, IP_EVENT_ETH_GOT_IP, &eth_event_handler, NULL));

    /* MAC config — internal EMAC on ESP32-P4 */
    eth_esp32_emac_config_t emac_config = ETH_ESP32_EMAC_DEFAULT_CONFIG();
    /* ESP-IDF v5.5: smi_gpio replaces the old smi_mdc/mdio_gpio_num fields */
    emac_config.smi_gpio.mdc_num  = ETH_PHY_MDC_GPIO;
    emac_config.smi_gpio.mdio_num = ETH_PHY_MDIO_GPIO;
    /* ESP32-P4-ETH: use EMAC_CLK_EXT_IN (external clock input, no GPIO enum needed) */
    emac_config.interface = EMAC_DATA_INTERFACE_RMII;

    eth_mac_config_t mac_config = ETH_MAC_DEFAULT_CONFIG();
    esp_eth_mac_t *mac = esp_eth_mac_new_esp32(&emac_config, &mac_config);
    if (mac == NULL) {
        ESP_LOGE(TAG, "Failed to create Ethernet MAC");
        return false;
    }

    /* PHY config */
    eth_phy_config_t phy_config = ETH_PHY_DEFAULT_CONFIG();
    phy_config.phy_addr    = ETH_PHY_ADDR;
    phy_config.reset_gpio_num = ETH_PHY_RST_GPIO;

    esp_eth_phy_t *phy = NULL;
#if ETH_PHY_TYPE == ETH_PHY_IP101
    phy = esp_eth_phy_new_ip101(&phy_config);
#elif ETH_PHY_TYPE == ETH_PHY_LAN8720
    phy = esp_eth_phy_new_lan87xx(&phy_config);
#elif ETH_PHY_TYPE == ETH_PHY_RTL8201
    phy = esp_eth_phy_new_rtl8201(&phy_config);
#else
    #error "Unknown PHY type — set ETH_PHY_TYPE to ETH_PHY_IP101, ETH_PHY_LAN8720, or ETH_PHY_RTL8201"
#endif

    if (phy == NULL) {
        ESP_LOGE(TAG, "Failed to create Ethernet PHY");
        return false;
    }

    /* Install Ethernet driver */
    esp_eth_config_t eth_config = ETH_DEFAULT_CONFIG(mac, phy);
    esp_eth_handle_t eth_handle = NULL;
    ESP_ERROR_CHECK(esp_eth_driver_install(&eth_config, &eth_handle));

    /* Attach netif glue */
    esp_eth_netif_glue_handle_t glue = esp_eth_new_netif_glue(eth_handle);
    ESP_ERROR_CHECK(esp_netif_attach(eth_netif, glue));

    /* Start Ethernet */
    ESP_ERROR_CHECK(esp_eth_start(eth_handle));

    ESP_LOGI(TAG, "Waiting for Ethernet + DHCP (15 s timeout)...");
    EventBits_t bits = xEventGroupWaitBits(
        s_eth_event_group,
        ETH_CONNECTED_BIT,
        pdFALSE,
        pdFALSE,
        pdMS_TO_TICKS(15000));

    if (bits & ETH_CONNECTED_BIT) {
        ESP_LOGI(TAG, "Ethernet ready");
        return true;
    }

    ESP_LOGE(TAG, "Ethernet timed out — check cable and PHY config");
    return false;
}

/* ── MQTT init ───────────────────────────────────────────────────────────── */
static void mqtt_init(void)
{
    esp_mqtt_client_config_t mqtt_cfg = {
        .broker.address.uri = MQTT_BROKER_URI,
    };
    s_mqtt_client = esp_mqtt_client_init(&mqtt_cfg);
    ESP_ERROR_CHECK(esp_mqtt_client_register_event(
        s_mqtt_client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL));
    ESP_ERROR_CHECK(esp_mqtt_client_start(s_mqtt_client));
    ESP_LOGI(TAG, "MQTT client started → %s", MQTT_BROKER_URI);
}

/* ── UI builder ──────────────────────────────────────────────────────────── */
static lv_obj_t *create_color_button(lv_obj_t *parent,
                                     const color_button_t *cfg,
                                     int x, int y)
{
    lv_obj_t *btn = lv_button_create(parent);
    lv_obj_set_size(btn, 210, 100);
    lv_obj_align(btn, LV_ALIGN_CENTER, x, y);
    lv_obj_set_style_bg_color(btn, lv_color_hex(cfg->color_hex),   LV_PART_MAIN);
    lv_obj_set_style_bg_color(btn, lv_color_hex(cfg->pressed_hex), LV_STATE_PRESSED);
    lv_obj_set_style_radius(btn, 18, LV_PART_MAIN);
    lv_obj_set_style_shadow_width(btn, 12, LV_PART_MAIN);
    lv_obj_set_style_shadow_opa(btn, LV_OPA_30, LV_PART_MAIN);
    lv_obj_add_event_cb(btn, color_btn_event_cb, LV_EVENT_PRESSED, (void *)cfg);

    lv_obj_t *label = lv_label_create(btn);
    lv_label_set_text(label, cfg->name);
    lv_obj_set_style_text_color(label, lv_color_hex(0xFFFFFF), LV_PART_MAIN);
    lv_obj_set_style_text_font(label, &lv_font_montserrat_14, LV_PART_MAIN);
    lv_obj_center(label);
    return btn;
}

static void create_test_ui(void)
{
    lv_obj_t *scr = lv_screen_active();
    lv_obj_set_style_bg_color(scr, lv_color_hex(0x2C3E50), LV_PART_MAIN);

    lv_obj_t *title = lv_label_create(scr);
    lv_label_set_text(title, "ESP32-P4-ETH  Ethernet MQTT Publisher");
    lv_obj_set_style_text_color(title, lv_color_hex(0xFFFFFF), LV_PART_MAIN);
    lv_obj_set_style_text_align(title, LV_TEXT_ALIGN_CENTER, LV_PART_MAIN);
    lv_obj_set_style_text_font(title, &lv_font_montserrat_14, LV_PART_MAIN);
    lv_obj_align(title, LV_ALIGN_TOP_MID, 0, 45);

    lv_obj_t *subtitle = lv_label_create(scr);
    lv_label_set_text(subtitle, "Tap a color — ESP32 publishes to MQTT over Ethernet.");
    lv_obj_set_style_text_color(subtitle, lv_color_hex(0xD6EAF8), LV_PART_MAIN);
    lv_obj_set_style_text_align(subtitle, LV_TEXT_ALIGN_CENTER, LV_PART_MAIN);
    lv_obj_set_style_text_font(subtitle, &lv_font_montserrat_14, LV_PART_MAIN);
    lv_obj_align(subtitle, LV_ALIGN_TOP_MID, 0, 100);

    status_label = lv_label_create(scr);
    lv_label_set_text(status_label, "Status: Starting Ethernet...");
    lv_obj_set_style_text_color(status_label, lv_color_hex(0xFFFFFF), LV_PART_MAIN);
    lv_obj_set_style_text_align(status_label, LV_TEXT_ALIGN_CENTER, LV_PART_MAIN);
    lv_obj_set_style_text_font(status_label, &lv_font_montserrat_14, LV_PART_MAIN);
    lv_obj_align(status_label, LV_ALIGN_TOP_MID, 0, 160);

    last_sent_label = lv_label_create(scr);
    lv_label_set_text(last_sent_label, "Last sent: none");
    lv_obj_set_style_text_color(last_sent_label, lv_color_hex(0xF9E79F), LV_PART_MAIN);
    lv_obj_set_style_text_align(last_sent_label, LV_TEXT_ALIGN_CENTER, LV_PART_MAIN);
    lv_obj_set_style_text_font(last_sent_label, &lv_font_montserrat_14, LV_PART_MAIN);
    lv_obj_align(last_sent_label, LV_ALIGN_TOP_MID, 0, 205);

    create_color_button(scr, &color_buttons[0], -240, -180);
    create_color_button(scr, &color_buttons[1],    0, -180);
    create_color_button(scr, &color_buttons[2],  240, -180);
    create_color_button(scr, &color_buttons[3], -240,  -40);
    create_color_button(scr, &color_buttons[4],    0,  -40);
    create_color_button(scr, &color_buttons[5],  240,  -40);

    lv_obj_t *random_btn = lv_button_create(scr);
    lv_obj_set_size(random_btn, 520, 110);
    lv_obj_align(random_btn, LV_ALIGN_CENTER, 0, 145);
    lv_obj_set_style_bg_color(random_btn, lv_color_hex(0x34495E), LV_PART_MAIN);
    lv_obj_set_style_bg_color(random_btn, lv_color_hex(0x1C2833), LV_STATE_PRESSED);
    lv_obj_set_style_radius(random_btn, 20, LV_PART_MAIN);
    lv_obj_set_style_shadow_width(random_btn, 14, LV_PART_MAIN);
    lv_obj_set_style_shadow_opa(random_btn, LV_OPA_30, LV_PART_MAIN);
    lv_obj_add_event_cb(random_btn, random_btn_event_cb, LV_EVENT_PRESSED, NULL);

    lv_obj_t *random_label = lv_label_create(random_btn);
    lv_label_set_text(random_label, "SEND RANDOM STRING");
    lv_obj_set_style_text_color(random_label, lv_color_hex(0xFFFFFF), LV_PART_MAIN);
    lv_obj_set_style_text_font(random_label, &lv_font_montserrat_14, LV_PART_MAIN);
    lv_obj_center(random_label);

    lv_obj_t *footer = lv_label_create(scr);
    lv_label_set_text(footer, "Topics: esp32/color  |  esp32/random");
    lv_obj_set_style_text_color(footer, lv_color_hex(0xAAB7B8), LV_PART_MAIN);
    lv_obj_set_style_text_align(footer, LV_TEXT_ALIGN_CENTER, LV_PART_MAIN);
    lv_obj_set_style_text_font(footer, &lv_font_montserrat_14, LV_PART_MAIN);
    lv_obj_align(footer, LV_ALIGN_BOTTOM_MID, 0, -40);
}

/* ── Entry point ─────────────────────────────────────────────────────────── */
void app_main(void)
{
    ESP_LOGI(TAG, "Starting ESP32-P4-ETH MQTT publisher");

    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    /* Start display early so status updates are visible */
    lv_display_t *disp = bsp_display_start();
    if (disp == NULL) {
        ESP_LOGE(TAG, "bsp_display_start() failed");
        return;
    }
    ESP_ERROR_CHECK(bsp_display_brightness_init());
    ESP_ERROR_CHECK(bsp_display_backlight_on());
    ESP_ERROR_CHECK(bsp_display_brightness_set(100));

    bsp_display_lock(0);
    create_test_ui();
    bsp_display_unlock();

    if (eth_init()) {
        mqtt_init();
    } else {
        bsp_display_lock(0);
        set_status("Status: Ethernet FAILED — check cable + PHY config");
        bsp_display_unlock();
    }

    ESP_LOGI(TAG, "Setup complete. Tap buttons to publish MQTT over Ethernet.");

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}