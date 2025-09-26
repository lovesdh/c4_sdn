#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <signal.h>
#include <arpa/inet.h>
#include <errno.h>

#pragma pack(push, 4)
typedef struct {
    int32_t op;
    int64_t op1;
    int64_t op2;
} RQT_PDU;

typedef struct {
    int64_t res;
} REP_PDU;
#pragma pack(pop)

volatile sig_atomic_t sigint_flag = 0;

void handle_sigint(int sig) {
    printf("[srv] SIGINT is coming!\n");
    sigint_flag = 1;
}

int64_t calculate(int op, int64_t op1, int64_t op2) {
    switch (op) {
        case 0x00000001: return op1 + op2;
        case 0x00000002: return op1 - op2;
        case 0x00000004: return op1 * op2;
        case 0x00000008: return op2 != 0 ? op1 / op2 : 0;
        case 0x00000010: return op2 != 0 ? op1 % op2 : 0;
        default: return 0;
    }
}

void srv_biz(int connfd) {
    RQT_PDU rqt_pdu;
    REP_PDU rep_pdu;

    while (1) {
        if (read(connfd, &rqt_pdu, sizeof(RQT_PDU)) <= 0) {
            break;
        }

        int op = ntohl(rqt_pdu.op);
        int64_t op1 = be64toh(rqt_pdu.op1);
        int64_t op2 = be64toh(rqt_pdu.op2);
        int64_t res = calculate(op, op1, op2);

        rep_pdu.res = htobe64(res);
        write(connfd, &rep_pdu, sizeof(REP_PDU));

        char op_char;
        switch (op) {
            case 0x00000001: op_char = '+'; break;
            case 0x00000002: op_char = '-'; break;
            case 0x00000004: op_char = '*'; break;
            case 0x00000008: op_char = '/'; break;
            case 0x00000010: op_char = '%'; break;
            default: op_char = '?'; break;
        }

        printf("[rqt_res] %ld %c %ld = %ld\n", op1, op_char, op2, res);
    }

    close(connfd);
}

int main(int argc, char *argv[]) {
    int listenfd, connfd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len = sizeof(client_addr);

    if (argc != 3) {
        fprintf(stderr, "Usage: %s <IP> <PORT>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    struct sigaction sa;
    sa.sa_handler = handle_sigint;
    sa.sa_flags = 0;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGINT, &sa, NULL) == -1) {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }

    struct sigaction sa_pipe;
    sa_pipe.sa_handler = SIG_IGN;
    sa_pipe.sa_flags = 0;
    sigemptyset(&sa_pipe.sa_mask);
    if (sigaction(SIGPIPE, &sa_pipe, NULL) == -1) {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }

    if ((listenfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(argv[1]);
    server_addr.sin_port = htons(atoi(argv[2]));

    if (bind(listenfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind");
        close(listenfd);
        exit(EXIT_FAILURE);
    }

    if (listen(listenfd, 5) < 0) {
        perror("listen");
        close(listenfd);
        exit(EXIT_FAILURE);
    }

    printf("[srv] server[%s:%s] is initializing!\n", argv[1], argv[2]);

    while (!sigint_flag) {
        if ((connfd = accept(listenfd, (struct sockaddr*)&client_addr, &client_len)) < 0) {
            if (errno == EINTR) break;
            perror("accept");
            continue;
        }

        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, INET_ADDRSTRLEN);
        printf("[srv] client[%s:%d] is accepted!\n", client_ip, ntohs(client_addr.sin_port));

        srv_biz(connfd);
        printf("[srv] client[%s:%d] is closed!\n",client_ip, ntohs(client_addr.sin_port));

    }

    close(listenfd);
    printf("[srv] listenfd is closed!\n");
    printf("[srv] server is going to exit!\n");

    return 0;
}
