#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LLMM_ETH_TCP_PORT 8742

/* DHCP + TCP server for the same LLMHOST binary protocol. Returns 0 on start. */
int llmm_eth_start(void);

/* If a TCP client is connected, host I/O uses that socket instead of UART. */
int llmm_eth_client_connected(void);
int llmm_eth_recv(void *buffer, uint32_t bytes, int timeout_ms);
int llmm_eth_send(const void *buffer, size_t bytes);
void llmm_eth_drop_client(void);

uint32_t llmm_eth_ipv4(void);
void llmm_eth_ipv4_octets(uint8_t out[4]);

#ifdef __cplusplus
}
#endif
