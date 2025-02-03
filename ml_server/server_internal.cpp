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
    const char *ack = "Message Processed";

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

    int msg_ix = 0;
    while(1) {
        std::cout << "Awaiting message #" << msg_ix << std::endl;
        // Accepting incoming connections
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
            perror("accept");
            exit(EXIT_FAILURE);
        }
        // Reading from client and sending a response
        read(new_socket, buffer, 1024);
        std::cout << "Client: " << buffer << std::endl;

        // Execute inference in its own thread
        std::optional<std::string> result = std::nullopt;
        std::future<std::optional<std::string>> fp = std::async(std::launch::async, infer, msg_ix, buffer);
        if(fp.valid()) {
            result = fp.get();
            if (result.has_value()) {
                // Print the resultant string
                std::string result_string = result.value();
                std::cout << "Return value from async thread is => " << result_string << std::endl;

                // Socket wants a character array sent
                char result_char[1024];
                strcpy(result_char, result_string.c_str());
                result_char[sizeof(result_char) - 1] = 0; // Truncate the result
                send(new_socket, result_char, strlen(result_char), 0);
            } else {
                std::cout << "Async thread returned null result";
            }
        }
        msg_ix++;
    }
    // Closing the socket
    close(new_socket);
    close(server_fd);
}
