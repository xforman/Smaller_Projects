#include "archive_fromat.h"


void parse_name(char* filepath, char* archive_header)
{
    char *end = filepath + strlen(filepath), *last_dir = NULL;
    unsigned i = 0;
    for (; i < 99 && end - i >= filepath; i++) {
        if (*(end - i) == '/')
            last_dir = end - i;
    }
    if (last_dir != end) {
        *last_dir = 0;
        strcpy(last_dir + 1, archive_header + 345);
    }
    strcpy(archive_header, filepath);
}

char *int_to_okt(int src)
{
    char *okt = malloc(8);
    memset(okt, 0, 8);
    for (int i = 0; i < 7 && src; i++, src/=8)
        okt[i] = src % 8;
    return okt;
}
