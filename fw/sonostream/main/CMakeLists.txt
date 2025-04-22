set(srcs "sonostream.c")

idf_component_register(SRCS "${srcs}"
                    PRIV_REQUIRES esp_driver_i2s esp_driver_gpio esp_driver_uart esp_wifi nvs_flash esp_driver_gptimer
                    INCLUDE_DIRS ".")
