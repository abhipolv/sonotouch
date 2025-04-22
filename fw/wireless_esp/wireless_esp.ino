#include<WiFi.h>

// WiFi network info:
static const char* ssid = "ASUS";
static const char* password = "jellybeanpop";

// Server info:
// static const char* host = "192.168.4.1";  // default IP for server ESP
static const char* host = "192.168.50.56";  // IPv4, AJA
static const uint16_t port = 80;

// Initializing WiFi connection
void setup() {
  Serial.begin(9600);
  WiFi.begin(ssid, password);

  // Wait until connected to WiFi
  while(WiFi.status() != WL_CONNECTED)
    Serial.println("Connecting...");
  Serial.println("Connected!");

  // Confirm WiFi client works
  WiFiClient client;
  client.connect(host, port);
  client.println("Client started.");
  client.stop();
}

void loop() {
  // Write to server
  WiFiClient client;
  client.connect(host, port);
  client.println("hihihi");
  client.stop();
}
