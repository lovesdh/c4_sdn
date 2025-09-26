#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <unistd.h>
#include <arpa/inet.h>

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

int get_operator_code(const char* op_str) {
    if (strncmp(op_str, "ADD", 3) == 0) return 0x00000001;
    if (strncmp(op_str, "SUB", 3) == 0) return 0x00000002;
    if (strncmp(op_str, "MUL", 3) == 0) return 0x00000004;
    if (strncmp(op_str, "DIV", 3) == 0) return 0x00000008;
    if (strncmp(op_str, "MOD", 3) == 0) return 0x00000010;
    return -1;
}

void send_request(int connfd, int32_t op, int64_t op1, int64_t op2) {
    RQT_PDU rqt_pdu;
    REP_PDU rep_pdu;

    rqt_pdu.op = htonl(op);
    rqt_pdu.op1 = htobe64(op1);
    rqt_pdu.op2 = htobe64(op2);

    write(connfd, &rqt_pdu, sizeof(RQT_PDU));
    read(connfd, &rep_pdu, sizeof(REP_PDU));

    rep_pdu.res = be64toh(rep_pdu.res);

    char op_char;
    switch (op) {
        case 0x00000001: op_char = '+'; break;
        case 0x00000002: op_char = '-'; break;
        case 0x00000004: op_char = '*'; break;
        case 0x00000008: op_char = '/'; break;
        case 0x00000010: op_char = '%'; break;
        default: op_char = '?'; break;
    }

    printf("[rep_rcv] %" PRId64 " %c %" PRId64 " = %" PRId64 "\n", op1, op_char, op2, rep_pdu.res);
}

void cli_biz(int connfd) {
    char command[128];
    char op_str[4];
    int32_t op;
    int64_t op1, op2;

    while (fgets(command, sizeof(command), stdin)) {
    	if (strncmp(command, "EXIT", 4) == 0) {
            printf("[cli] command EXIT received\n");
            break;
        }
        if (sscanf(command, "%s %" PRId64 " %" PRId64 "", op_str, &op1, &op2) == 3) {
            op = get_operator_code(op_str);
            if (op != -1) {
                send_request(connfd, op, op1, op2);
            } else {
                fprintf(stderr, "Unknown operation: %s\n", op_str);
            }
        } else {
            fprintf(stderr, "Invalid command format\n");
        }
    }

    close(connfd);
    printf("[cli] connfd is closed!\n");
    printf("[cli] client is going to exit!\n");
}

int main(int argc, char *argv[]) {
    int sockfd;
    struct sockaddr_in server_addr;

    if (argc != 3) {
        fprintf(stderr, "Usage: %s <IP> <PORT>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(atoi(argv[2]));

    if (inet_pton(AF_INET, argv[1], &server_addr.sin_addr) <= 0) {
        perror("inet_pton");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    if (connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("connect");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    printf("[cli] server[%s:%s] is connected!\n", argv[1], argv[2]);

    cli_biz(sockfd);

    return 0;
}
