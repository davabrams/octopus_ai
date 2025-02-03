#include <future>
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <string.h>
#include <optional>
#include "model_inference.cpp"

void server_loop() {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    char buffer[1024] = {0};
    const char *ack = "Message Acknowledged";

    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // Forcefully attaching socket to the port 8080
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(8080);

    // Binding the socket to the address and port
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    // Listening for connections
    if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    while(1) {
        // Accepting incoming connections
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
            perror("accept");
            exit(EXIT_FAILURE);
        }
        // Reading from client and sending a response
        read(new_socket, buffer, 1024);
        std::cout << "Client: " << buffer << std::endl;

        // Execute inference in its own thread
        std::future<std::optional<std::string>> fp = std::async(std::launch::async, infer, buffer);
        if(fp.valid()) {
            std::optional<std::string> result = fp.get();
            if (result.has_value()) {
                std::cout << "Return value from async thread is => " << result.value() << std::endl;
            } else {
                std::cout << "Async thread returned null result";
            }
        }

        std::cout << "Sending ACK to client...";
        send(new_socket, ack, strlen(ack), 0);
        std::cout << "Done." << std::endl;
    }
    // Closing the socket
    close(new_socket);
    close(server_fd);
}
