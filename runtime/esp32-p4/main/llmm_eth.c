#include "llmm_eth.h"

#include <errno.h>
#include <string.h>

#include "esp_eth.h"
#include "esp_eth_mac.h"
#include "esp_eth_phy.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "soc/soc_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lwip/def.h"
#include "lwip/inet.h"
#include "lwip/sockets.h"

#define LLMM_ETH_PHY_ADDR 1
#define LLMM_ETH_MDC_GPIO 31
#define LLMM_ETH_MDIO_GPIO 52
#define LLMM_ETH_RST_GPIO 51
#define LLMM_ETH_CLK_GPIO 50
#define LLMM_ETH_TX_EN_GPIO 49
#define LLMM_ETH_TXD0_GPIO 34
#define LLMM_ETH_TXD1_GPIO 35
#define LLMM_ETH_CRS_DV_GPIO 28
#define LLMM_ETH_RXD0_GPIO 29
#define LLMM_ETH_RXD1_GPIO 30

static const char *TAG = "llmm-eth";
static volatile int s_client = -1;
static volatile uint32_t s_ip = 0;

void llmm_eth_drop_client(void)
{
    int fd = s_client;
    s_client = -1;
    if (fd >= 0) {
        shutdown(fd, SHUT_RDWR);
        close(fd);
    }
}

uint32_t llmm_eth_ipv4(void)
{
    return s_ip;
}

void llmm_eth_ipv4_octets(uint8_t out[4])
{
    uint32_t host = lwip_ntohl(s_ip);
    out[0] = (uint8_t)((host >> 24) & 0xFF);
    out[1] = (uint8_t)((host >> 16) & 0xFF);
    out[2] = (uint8_t)((host >> 8) & 0xFF);
    out[3] = (uint8_t)(host & 0xFF);
}

int llmm_eth_client_connected(void)
{
    return s_client >= 0;
}

int llmm_eth_recv(void *buffer, uint32_t bytes, int timeout_ms)
{
    int fd = s_client;
    if (fd < 0) return -1;
    struct timeval tv;
    if (timeout_ms < 0) {
        /* Blocking host reads: never use a 0 timeout (that looks like a peer close). */
        tv.tv_sec = 60;
        tv.tv_usec = 0;
    } else {
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
    }
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    int n = recv(fd, buffer, bytes, 0);
    if (n == 0) {
        llmm_eth_drop_client();
        return -1;
    }
    if (n < 0) {
        if (errno == EWOULDBLOCK || errno == EAGAIN) return 0;
        llmm_eth_drop_client();
        return -1;
    }
    return n;
}

int llmm_eth_send(const void *buffer, size_t bytes)
{
    int fd = s_client;
    if (fd < 0) return -1;
    const uint8_t *cursor = buffer;
    while (bytes != 0) {
        int n = send(fd, cursor, bytes, 0);
        if (n <= 0) {
            llmm_eth_drop_client();
            return -1;
        }
        cursor += n;
        bytes -= (size_t)n;
    }
    return (int)(cursor - (const uint8_t *)buffer);
}

static void got_ip_handler(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    (void)arg;
    (void)base;
    (void)id;
    ip_event_got_ip_t *event = data;
    s_ip = event->ip_info.ip.addr;
    ESP_LOGI(TAG, "got IP " IPSTR " — TCP host :%d", IP2STR(&event->ip_info.ip), LLMM_ETH_TCP_PORT);
}

static void tcp_server_task(void *arg)
{
    (void)arg;
    int listen_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listen_fd < 0) {
        vTaskDelete(NULL);
        return;
    }
    int yes = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons(LLMM_ETH_TCP_PORT),
        .sin_addr.s_addr = htonl(INADDR_ANY),
    };
    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0 || listen(listen_fd, 1) != 0) {
        close(listen_fd);
        vTaskDelete(NULL);
        return;
    }
    ESP_LOGI(TAG, "listening on TCP %d", LLMM_ETH_TCP_PORT);
    for (;;) {
        struct sockaddr_in peer;
        socklen_t peer_len = sizeof(peer);
        int client = accept(listen_fd, (struct sockaddr *)&peer, &peer_len);
        if (client < 0) {
            vTaskDelay(pdMS_TO_TICKS(200));
            continue;
        }
        setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
        int prev = s_client;
        s_client = client;
        if (prev >= 0 && prev != client) {
            shutdown(prev, SHUT_RDWR);
            close(prev);
        }
        ESP_LOGI(TAG, "client %s:%d", inet_ntoa(peer.sin_addr), (int)ntohs(peer.sin_port));
    }
}

static void eth_ip_fallback_task(void *arg);

int llmm_eth_start(void)
{
    if (esp_netif_init() != ESP_OK) return -10;
    if (esp_event_loop_create_default() != ESP_OK) return -11;

    eth_mac_config_t mac_config = ETH_MAC_DEFAULT_CONFIG();
    eth_phy_config_t phy_config = ETH_PHY_DEFAULT_CONFIG();
    phy_config.phy_addr = LLMM_ETH_PHY_ADDR;
    phy_config.reset_gpio_num = LLMM_ETH_RST_GPIO;

    eth_esp32_emac_config_t emac = ETH_ESP32_EMAC_DEFAULT_CONFIG();
    emac.smi_gpio.mdc_num = LLMM_ETH_MDC_GPIO;
    emac.smi_gpio.mdio_num = LLMM_ETH_MDIO_GPIO;
    emac.interface = EMAC_DATA_INTERFACE_RMII;
    emac.clock_config.rmii.clock_mode = EMAC_CLK_EXT_IN;
    emac.clock_config.rmii.clock_gpio = LLMM_ETH_CLK_GPIO;
#if SOC_EMAC_USE_MULTI_IO_MUX
    emac.emac_dataif_gpio.rmii.tx_en_num = LLMM_ETH_TX_EN_GPIO;
    emac.emac_dataif_gpio.rmii.txd0_num = LLMM_ETH_TXD0_GPIO;
    emac.emac_dataif_gpio.rmii.txd1_num = LLMM_ETH_TXD1_GPIO;
    emac.emac_dataif_gpio.rmii.crs_dv_num = LLMM_ETH_CRS_DV_GPIO;
    emac.emac_dataif_gpio.rmii.rxd0_num = LLMM_ETH_RXD0_GPIO;
    emac.emac_dataif_gpio.rmii.rxd1_num = LLMM_ETH_RXD1_GPIO;
#endif

    esp_eth_mac_t *mac = esp_eth_mac_new_esp32(&emac, &mac_config);
    if (mac == NULL) return -1;
    esp_eth_phy_t *phy = esp_eth_phy_new_generic(&phy_config);
    if (phy == NULL) {
        mac->del(mac);
        return -2;
    }
    esp_eth_handle_t eth = NULL;
    esp_eth_config_t config = ETH_DEFAULT_CONFIG(mac, phy);
    if (esp_eth_driver_install(&config, &eth) != ESP_OK) {
        mac->del(mac);
        phy->del(phy);
        return -3;
    }

    esp_netif_config_t netif_cfg = ESP_NETIF_DEFAULT_ETH();
    esp_netif_t *netif = esp_netif_new(&netif_cfg);
    if (netif == NULL) return -12;
    if (esp_netif_attach(netif, esp_eth_new_netif_glue(eth)) != ESP_OK) return -13;
    if (esp_event_handler_register(IP_EVENT, IP_EVENT_ETH_GOT_IP, &got_ip_handler, NULL) != ESP_OK) return -14;
    if (esp_eth_start(eth) != ESP_OK) return -15;

    if (xTaskCreate(tcp_server_task, "llmm-tcp", 6144, NULL, 5, NULL) != pdPASS) return -4;
    /* Don't block UART: wait for DHCP in a side task, then fall back to 192.168.4.1. */
    if (xTaskCreate(eth_ip_fallback_task, "llmm-eth-ip", 3072, netif, 4, NULL) != pdPASS) return -16;
    return 0;
}

static void eth_ip_fallback_task(void *arg)
{
    esp_netif_t *netif = arg;
    for (int i = 0; i < 150 && s_ip == 0; ++i) {
        vTaskDelay(pdMS_TO_TICKS(100));
    }
    if (s_ip == 0 && netif != NULL) {
        (void)esp_netif_dhcpc_stop(netif);
        esp_netif_ip_info_t info = {0};
        info.ip.addr = ESP_IP4TOADDR(192, 168, 4, 1);
        info.gw.addr = ESP_IP4TOADDR(192, 168, 4, 1);
        info.netmask.addr = ESP_IP4TOADDR(255, 255, 255, 0);
        if (esp_netif_set_ip_info(netif, &info) == ESP_OK) {
            (void)esp_netif_dhcps_start(netif);
            s_ip = info.ip.addr;
            ESP_LOGI(TAG, "no DHCP lease; static " IPSTR " + DHCP server", IP2STR(&info.ip));
        }
    }
    vTaskDelete(NULL);
}
