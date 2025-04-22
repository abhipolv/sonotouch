#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "freertos/stream_buffer.h"
#include "soc/soc_caps.h"
#include "driver/i2s_pdm.h"
#include "driver/uart.h"
#include "driver/gpio.h"
#include "driver/gptimer.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "nvs_flash.h"
#include "sdkconfig.h"

#define VPU_PDM_RX_CLK_IO               GPIO_NUM_5
#define VPU_PDM_RX_DIN1_IO              GPIO_NUM_1

#if SOC_I2S_PDM_MAX_RX_LINES > 1
#define VPU_PDM_RX_DIN2_IO              GPIO_NUM_2
#endif /* SOC_I2S_PDM_MAX_RX_LINES */ 

#if SOC_I2S_SUPPORTS_PDM2PCM 
#define PDM_RX_FREQ_HZ          44100
#else
#define PDM_RX_FREQ_HZ          2048000
#endif /* SOC_I2S_SUPPORTS_PDM2PCM */ 

#define WIFI_EN				1
#define PRINT_EN			0
#define SERIAL_EN			0
#define DIAG_EN				0
#define RECV_EN				0

#define DUAL_MIC     			0
#define BUFF_SIZE			4096
#define ITEM_SIZE			sizeof(uint16_t)
#define PTR_SIZE			sizeof(uint16_t *)
#define BUFF_IN_BYTES			BUFF_SIZE * ITEM_SIZE

DMA_ATTR uint8_t *rx_buffer = NULL;

static const char *TAG = "Sonostream";
i2s_chan_handle_t vpu_handle;

#if PRINT_EN
static inline void print_buffer_state(uint16_t *buf, char *header, const int peak) {
    printf("\n");
    printf("BEGIN %s -------------------------\n", header);
    for(int i = 0;  i < peak; i++) {
	printf("%d ", buf[BUFF_SIZE/2]);
    }
    printf(". . . ");
    for(int i = peak;  i > 0; i--) {
	printf("%d ", buf[BUFF_SIZE - i]);
    }
    printf("\n");
    printf("END %s -------------------------\n", header);
    printf("\n");
} 
#endif /* PRINT_EN */

#if SERIAL_EN
void uart_init() {
    const uart_port_t uart_port = UART_NUM_0;
    uart_config_t uart_config = {
	    .baud_rate = 921600,
	    .data_bits = UART_DATA_8_BITS,
	    .parity = UART_PARITY_DISABLE,
	    .stop_bits = UART_STOP_BITS_1,
	    .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
    };
    uart_param_config(UART_NUM_0, &uart_config);
    int intr_alloc_flags = 0;

    #if CONFIG_UART_ISR_IN_IRAM
	intr_alloc_flags = ESP_INTR_FLAG_IRAM;
    #endif /* CONFIG_UART_ISR_IN_IRAM */

    ESP_ERROR_CHECK(uart_driver_install(uart_port, BUFF_SIZE, 0, 0, NULL, intr_alloc_flags));
    ESP_ERROR_CHECK(uart_param_config(uart_port, &uart_config));
}
#endif /* SERIAL_EN */

#if WIFI_EN
#define WIFI_SSID      "definitelynotarouter"
#define WIFI_PASS      "notapwd777"
#define TCP_PORT       3333
#define HOST_IP	       "192.168.50.155"

#define QUEUE_SIZE			10
#define MAXIMUM_CONNECTION_RETRY 	10
#define WIFI_CONNECTED_BIT 		BIT0
#define WIFI_FAIL_BIT      		BIT1

DMA_ATTR uint8_t *tx_buffer = NULL;
QueueHandle_t vpu_queue_handle = NULL;
volatile uint8_t queue_count = 0;
volatile uint8_t socket_started = 0;

static EventGroupHandle_t s_wifi_event_group;
static int s_retry_num = 0;

static void event_handler(void* arg, esp_event_base_t event_base, int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
	esp_wifi_connect();
    }
    else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
	if (s_retry_num < MAXIMUM_CONNECTION_RETRY) {
	    esp_wifi_connect();
	    s_retry_num++;
	    ESP_LOGI(TAG, "Attempting to connect to the AP...");
	}
	else {
	    xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
	}
	ESP_LOGI(TAG,"Connection to the AP fail");
    }
    else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
	ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
	ESP_LOGI(TAG, "got ip:" IPSTR, IP2STR(&event->ip_info.ip));
	s_retry_num = 0;
	xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

static void wifi_init_sta(void) {
    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());

    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                        ESP_EVENT_ANY_ID,
                                                        &event_handler,
                                                        NULL,
                                                        &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                                        IP_EVENT_STA_GOT_IP,
                                                        &event_handler,
                                                        NULL,
                                                        &instance_got_ip));
    wifi_config_t sta_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASS
        }
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta_config));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_LOGI(TAG, "Device successfully initialized as a WiFi station!");

    EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group,
            WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
            pdFALSE,
            pdFALSE,
            portMAX_DELAY);

    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "Connected to ap SSID:%s password:%s",
                 WIFI_SSID, WIFI_PASS);
    } else if (bits & WIFI_FAIL_BIT) {
        ESP_LOGI(TAG, "Failed to connect to SSID:%s, password:%s",
                 WIFI_SSID, WIFI_PASS);
    } else {
        ESP_LOGE(TAG, "UNEXPECTED EVENT");
    }

    ESP_ERROR_CHECK(esp_event_handler_instance_unregister(IP_EVENT, IP_EVENT_STA_GOT_IP, instance_got_ip));
    ESP_ERROR_CHECK(esp_event_handler_instance_unregister(WIFI_EVENT, ESP_EVENT_ANY_ID, instance_any_id));
    vEventGroupDelete(s_wifi_event_group);
}

static inline int socket_send(const int sock, const uint8_t * data, const size_t len) {
    int to_write = len;
    while (to_write > 0) {
        int written = send(sock, data + (len - to_write), to_write, 0);
        if (written < 0 && errno != EINPROGRESS && errno != EAGAIN && errno != EWOULDBLOCK) {
            ESP_LOGE(TAG, "Error occurred during sending a chunk: errno %d\n", errno);
            return -1;
        }
        if (written > -1) {
            to_write -= written;
        }
    }
    return len;
}

void socket_tx_task(void *args) {
    char host_ip[] = HOST_IP;
    int addr_family = 0;
    int ip_protocol = 0;

    #if RECV_EN
    char rec_buffer[128];
    #endif /* RECV_EN */

    #if DIAG_EN
    gptimer_handle_t socket_tx_timer = NULL;
    gptimer_config_t socket_tx_timer_config = {
	.clk_src = GPTIMER_CLK_SRC_DEFAULT,
	.direction = GPTIMER_COUNT_UP,
	.resolution_hz = 1 * 1000 * 1000, /* 1MHz, 1 tick = 1us */
    };
    ESP_ERROR_CHECK(gptimer_new_timer(&socket_tx_timer_config, &socket_tx_timer));
    ESP_ERROR_CHECK(gptimer_enable(socket_tx_timer));
    ESP_ERROR_CHECK(gptimer_start(socket_tx_timer));
    #endif /* DIAG_EN */

    while (1) {
        struct sockaddr_in dest_addr;
        inet_pton(AF_INET, host_ip, &dest_addr.sin_addr);
        dest_addr.sin_family = AF_INET;
        dest_addr.sin_port = htons(TCP_PORT);
        addr_family = AF_INET;
        ip_protocol = IPPROTO_IP;

        int sock =  socket(addr_family, SOCK_STREAM, ip_protocol);
        if (sock < 0) {
            ESP_LOGE(TAG, "Unable to create socket: errno %d", errno);
            break;
        }

        ESP_LOGI(TAG, "Socket created, connecting to %s:%d", host_ip, TCP_PORT);

	fcntl(sock, F_SETFL, fcntl( sock, F_GETFL, 0 ) | O_NONBLOCK);
	int nodelay = 1;
	setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (void *)&nodelay, sizeof(int));

	while (1) {
	    int err = connect(sock, (struct sockaddr *)&dest_addr, sizeof(dest_addr));
	    if (err < 0 && errno == EINPROGRESS) {
		continue;
	    }
	    socket_started = 1;
	    ESP_LOGI(TAG, "Successfully connected");

	    while (1) {
		if(xQueueReceive(vpu_queue_handle, &tx_buffer, (TickType_t) 0) == pdPASS) {

		    #if PRINT_EN
		    print_buffer_state(rx_buffer, "Pop Queue", 8);
		    #endif /* PRINT_EN */

		    #if DIAG_EN
		    ESP_ERROR_CHECK(gptimer_set_raw_count(socket_tx_timer, 0));
		    #endif /* DIAG_EN */

		    int sent = socket_send(sock, (uint8_t *)tx_buffer, BUFF_IN_BYTES);
		    if (sent < 0) {
			ESP_LOGE(TAG, "Error occurred during sending a frame: errno %d", errno);
			break;
		    }
		    else {
			ESP_LOGI(TAG, "Success: %d bytes sent", sent);
		    }

		    #if DIAG_EN
		    uint64_t elapsed_time_us = 0;
		    ESP_ERROR_CHECK(gptimer_get_raw_count(socket_tx_timer, &elapsed_time_us));
		    ESP_LOGI(TAG, "Socket Tx duration: %llu us", elapsed_time_us);
		    queue_count = (uint8_t)(uxQueueMessagesWaiting(vpu_queue_handle));
		    ESP_LOGI(TAG, "Socket task removed an element, current size: %d", queue_count);
		    #endif /* DIAG_EN */

		    #if RECV_EN
		    int received = recv(sock, rec_buffer, sizeof(rec_buffer) - 1, 0);
		    if (received < 0) {
			ESP_LOGE(TAG, "recv failed: errno %d", errno);
			break;
		    }
		    else {
			rec_buffer[received] = 0;
			ESP_LOGI(TAG, "Received %d bytes from %s:", received, host_ip);
			ESP_LOGI(TAG, "%s", rec_buffer);
		    }
		    #endif /* RECV_EN */
		}
	    }
	}
	if (sock != -1) {
	    ESP_LOGE(TAG, "Shutting down socket and restarting...");
	    shutdown(sock, 0);
	    close(sock);
	}
    }
    vTaskDelete(NULL);
    #if DIAG_EN
    ESP_ERROR_CHECK(gptimer_stop(socket_tx_timer));
    ESP_ERROR_CHECK(gptimer_disable(socket_tx_timer));
    ESP_ERROR_CHECK(gptimer_del_timer(socket_tx_timer));
    #endif /* DIAG_EN */
}
#endif /* WIFI_EN */

static void vpu_init(void) {

    #if SOC_I2S_SUPPORTS_PDM2PCM
	ESP_LOGI(TAG, "Receiving data in PCM format");
    #else
	ESP_LOGIDUAL(TAG, "Receiving data in raw PDM format");
    #endif  /* SOC_I2S_SUPPORTS_PDM2PCM */

    i2s_chan_config_t vpu_rx_chan_cfg = {
	.id = I2S_NUM_AUTO,
	.role = I2S_ROLE_MASTER,
	.dma_desc_num = 6,
	.dma_frame_num = 512,
	.auto_clear_after_cb = false,
	.auto_clear_before_cb = false,
	.allow_pd = false,
	.intr_priority = 0,
    };

    ESP_ERROR_CHECK(i2s_new_channel(&vpu_rx_chan_cfg, NULL, &vpu_handle));

    i2s_pdm_rx_config_t vpu_rx_cfg = {
	.clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(PDM_RX_FREQ_HZ),

	#if SOC_I2S_SUPPORTS_PDM2PCM

	#if SOC_I2S_PDM_MAX_RX_LINES > 1 && DUAL_MIC

	.slot_cfg = {
	    .data_bit_width = I2S_DATA_BIT_WIDTH_16BIT,
	    .slot_bit_width = I2S_SLOT_BIT_WIDTH_AUTO,
	    .slot_mode = I2S_SLOT_MODE_STEREO,
	    .slot_mask = I2S_PDM_RX_LINE0_SLOT_RIGHT | I2S_PDM_RX_LINE1_SLOT_LEFT,
	    .data_fmt = I2S_PDM_DATA_FMT_PCM ,

	#if SOC_I2S_SUPPORTS_PDM_RX_HP_FILTER
	    .hp_en = true,
	    .hp_cut_off_freq_hz = 35.5,
	    .amplify_num = 1,       
	#endif  /* SOC_I2S_SUPPORTS_PDM_RX_HP_FILTER */

	},
	#else
	.slot_cfg = I2S_PDM_RX_SLOT_PCM_FMT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
	#endif /* SOC_I2S_PDM_MAX_RX_LINES > 1 && DUAL_MIC */ 

    #else
	.slot_cfg = I2S_PDM_RX_SLOT_RAW_FMT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
    #endif  /* SOC_I2S_SUPPORTS_PDM2PCM */
	.gpio_cfg = {
	    .clk = VPU_PDM_RX_CLK_IO, 
    #if SOC_I2S_PDM_MAX_RX_LINES > 1 && DUAL_MIC
	    .dins = {
	    VPU_PDM_RX_DIN1_IO,              
	    VPU_PDM_RX_DIN2_IO,              
	    },
    #else
	    .din = VPU_PDM_RX_DIN1_IO,              
    #endif /* SOC_I2S_PDM_MAX_RX_LINES > 1 && DUAL_MIC */
	    .invert_flags = {
		.clk_inv = false,
	    },
	},
    };

    ESP_ERROR_CHECK(i2s_channel_init_pdm_rx_mode(vpu_handle, &vpu_rx_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(vpu_handle));
}

void vpu_read_pdm_rx_task(void *args) {
    size_t r_bytes = 0;

    #if DIAG_EN
    gptimer_handle_t pdm_rx_timer = NULL;
    gptimer_config_t pdm_rx_timer_config = {
	.clk_src = GPTIMER_CLK_SRC_DEFAULT,
	.direction = GPTIMER_COUNT_UP,
	.resolution_hz = 1 * 1000 * 1000, /* 1MHz, 1 tick = 1us */
    };
    ESP_ERROR_CHECK(gptimer_new_timer(&pdm_rx_timer_config, &pdm_rx_timer));
    ESP_ERROR_CHECK(gptimer_enable(pdm_rx_timer));
    ESP_ERROR_CHECK(gptimer_start(pdm_rx_timer));
    #endif /* DIAG_EN */

    while (1) {
	#if DIAG_EN
	ESP_ERROR_CHECK(gptimer_set_raw_count(pdm_rx_timer, 0));
	#endif /* DIAG_EN */

	if (i2s_channel_read(vpu_handle, rx_buffer, BUFF_IN_BYTES, &r_bytes, 1000) == ESP_OK) {

	    #if WIFI_EN
	    xQueueSendToBack( vpu_queue_handle, &rx_buffer, ( TickType_t ) 0 );
	    #endif /* WIFI_EN */

	    #if SERIAL_EN	
	    uart_write_bytes(UART_NUM_0, (const char *)rx_buffer, r_bytes);
	    #endif /* SERIAL_EN */

	    #if PRINT_EN
	    print_buffer_state(rx_buffer, "Push Queue", 8);
	    #endif /* PRINT_EN */

	    #if DIAG_EN
	    uint64_t elapsed_time_us = 0;
	    ESP_ERROR_CHECK(gptimer_get_raw_count(pdm_rx_timer, &elapsed_time_us));
	    ESP_LOGI(TAG, "PDM Rx duration: %llu us", elapsed_time_us);
	    queue_count = (uint8_t)(uxQueueMessagesWaiting(vpu_queue_handle));
	    ESP_LOGI(TAG, "PDM task added an element, current size: %d", queue_count);
	    #endif /* DIAG_EN */
	}
	else {
	  ESP_LOGE(TAG, "Read Task: i2s read failed\n");
	}
    }
    vTaskDelete(NULL);
    #if DIAG_EN
    ESP_ERROR_CHECK(gptimer_stop(pdm_rx_timer));
    ESP_ERROR_CHECK(gptimer_disable(pdm_rx_timer));
    ESP_ERROR_CHECK(gptimer_del_timer(pdm_rx_timer));
    #endif /* DIAG_EN */
}

void app_main(void) {

    rx_buffer = (uint8_t*)calloc(1, BUFF_IN_BYTES);
    assert(rx_buffer);
    
    vpu_init();

    #if SERIAL_EN
    uart_init();
    #endif /* SERIAL_EN */	

    #if WIFI_EN
    tx_buffer = (uint8_t*)calloc(1, BUFF_IN_BYTES);
    assert(tx_buffer);

    vpu_queue_handle = xQueueCreate(QUEUE_SIZE, PTR_SIZE);
    assert(vpu_queue_handle);

    esp_err_t nvs_ret = nvs_flash_init();
    if (nvs_ret == ESP_ERR_NVS_NO_FREE_PAGES || nvs_ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
      ESP_ERROR_CHECK(nvs_flash_erase());
      nvs_ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(nvs_ret);

    wifi_init_sta();

    xTaskCreatePinnedToCore(socket_tx_task, "socket_tx_task", 4096, NULL, configMAX_PRIORITIES - 1, NULL, PRO_CPU_NUM);    
    #endif /* WIFI_EN */

    xTaskCreatePinnedToCore(vpu_read_pdm_rx_task, "vpu_read_pdm_rx_task", 4096, NULL, configMAX_PRIORITIES - 1, NULL, APP_CPU_NUM);
}
