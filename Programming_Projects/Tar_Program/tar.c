#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <grp.h>
#include <pwd.h>
#include <unistd.h>
#include <dirent.h>

char *int_to_okt(int, int);
bool parse_name(char *, char *);
void insert_number(int num, int max_length, char *header);
bool create_header(char *filepath, char *header, struct stat *);

char *int_to_okt(int src, int length)
{
    char *okt = malloc(length);
    memset(okt, 0, length);
    okt[0] = '0';
    for (int i = 0; i < length - 1 && src; i++, src/=8)
        okt[i] = src % 8;
    return okt;
}

bool parse_name(char* filepath, char* archive_header)
{
    char *end = filepath + strlen(filepath), *last_dir = end;
    unsigned i = 0;
    for (; i < 99 && end - i >= filepath; i++) {
        if (*(end - i) == '/')
            last_dir = end - i;
    }
    if (i == 99 && last_dir != end) {
        *last_dir = 0;
        strcpy(last_dir + 1, archive_header + 345);
    }
    if (i < 99 || last_dir != end)
        strcpy(archive_header, filepath);
    if (!*last_dir)
        *last_dir = '/';
    return i <= 99 && last_dir != end;
}

void insert_number(int num, int max_length, char *header)
{
    char *okt = int_to_okt(num, 8);
    strcpy(header + max_length - strlen(okt) - 1, okt);
    free(okt);
}

bool create_header(char *filepath, char *header, struct stat *file_stat)
{
    parse_name(filepath, header);
    if ((!S_ISREG(file_stat->st_mode) && !S_ISDIR(file_stat->st_mode)) ||
            !parse_name(filepath, header))
        return false;
    struct group *group = getgrnam(filepath);
    struct passwd *owner = getpwuid(filepath);
    insert_number(file_stat->st_mode & (unsigned) 511, 8, header + 100);
    insert_number(file_stat->st_uid, 8, header + 108);
    insert_number(file_stat->st_gid, 8, header + 116);
    insert_number(S_ISDIR(file_stat->st_mode) ? 0 : file_stat->st_size, 12, header + 124);
    insert_number(file_stat->st_mtime, 12, header + 136);
    header[156] = S_ISDIR(file_stat->st_mode) ? '5' : '0';
    if (owner->pw_name)
        strcpy(header + 265, owner->pw_name);
    if (group->gr_name)
        strcpy(header + 297, group->gr_name);

    unsigned checksum = 8 * ' ';  // the value of the control sum itself
    for (int i = 0; i < 512; i++) {
        checksum += (unsigned char) header[i];
    }
    insert_number(checksum, 7, header + 148);
    header[155] = ' ';
    return true;
}

bool output_file(char *path, int output)
{
    char input_cpy[512];
    ssize_t size = 0;
    int file = open(path, O_RDONLY);
    if (file < 0)
        return false;
    while ((size = read(file, input_cpy, 512)) == 512) {
        write(output, input_cpy, size);
        memset(input_cpy, 0, 512);
    }
    write(output, input_cpy, 512 - size);
    return true;
}

bool create_archive(char *path, int output, bool verbose)
{
    if (verbose)
        fprintf(stderr, "%s", path);
    char *header = malloc(512);
    struct stat *source_stat;
    stat(path, source_stat);
    memset(header, 0, 512);
    if (!create_header(path, header, source_stat)) {
        fprintf(stderr, "Name too long");
        free(header);
        return false;
    }
    write(output, header, 512);
    if (S_ISREG(source_stat->st_mode))
        return output_file(path, output);

    bool skipped = false;
    struct dirent **namelist;
    int files = scandir(path, &namelist, NULL, &alphasort);
    path = realloc(path, strlen(path) + 256);
    char *dir_name = path + strlen(path) + 1;
    memset(dir_name, 0, 256);
    for (int i = 0; i <= files; i++) {
        strncpy(dir_name, namelist[i]->d_name, 255);
        stat(path, source_stat);
        if (S_ISDIR(source_stat->st_mode))
            skipped |= !create_archive(path, output, verbose);
    }
    for (int i = 0; i <= files; i++) {
        strncpy(dir_name, namelist[i]->d_name, 255);
        stat(path, source_stat);
        if (S_ISREG(source_stat->st_mode))
            skipped |= !create_archive(path, output, verbose);
    }
    return !skipped;
}

int cmp_files(const void *file1, const void *file2)
{
    struct stat f1, f2;
    stat(*((char **) file1), &f1);
    stat(*((char **) file2), &f2);
    if (S_ISDIR(f1.st_mode) ^ S_ISDIR(f1.st_mode))
        return S_ISDIR(f1.st_mode) ? 1 : -1;
    return strcmp(*((char **) file1), *((char **) file2));
}

int main(int argc, char **argv)
{
    char **files = malloc(argc - 3);
    memcpy(files, argv + 3, argc - 3);
    qsort(files, argc - 3, sizeof(char **), &cmp_files);
    return 0;
}
