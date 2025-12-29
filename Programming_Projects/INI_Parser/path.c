#include "path.h"

enum cycle_t dplct_file(switch_t switches, filesystem_t *files, char *filename, list_t *sources)
{
    bool dplct = false;
    for (unsigned i = 0; i < files->size && !dplct; i++) {
        if (!strcmp(filename, files->files[i]))
            dplct = true;
    }
    size_t len = strlen(filename) + 1;
    char *new_file = malloc(len + 1);
    strcpy(new_file, filename);
    new_file[len - 1] = ' ';
    new_file[len] = 0;
    char *is_cycle = strstr(sources->list, new_file);

    if ((!dplct || !switches.guard) && !is_cycle) {
        insert_file(files, filename);
        sources->len += len;
        sources->list = realloc(sources->list, sources->len + 1);
        strcat(sources->list, new_file);
    } else {
        if (is_cycle && switches.strict)
            fprintf(stderr, "Error: cycle detected in: \n%s\n", filename);
        free(filename);
    }
    free(new_file);
    return is_cycle ? cycle : (dplct ? duplicate : new);
}

void insert_file(filesystem_t *files, char *filename)
{
    files->size++;
    files->files = (char **) realloc(files->files, sizeof(void *) * (files->size));
    files->files[files->size - 1] = filename;
}

char *create_filepath(char *source, char *include)
{
    if (!include || !strchr(source, '/'))
        return include;
    int end = strlen(source);
    while (source[end - 1] != '/')
        end--;
    char *compl_path = malloc(strlen(source) + strlen(include) + 1);
    compl_path = strncpy(compl_path, source, end + 1);
    compl_path[end] = 0;
    strcat(compl_path, include);
    free(include);
    return compl_path;
}

/// checks if a dir is either './' or '../' if yes then it recursively checks again and
/// then skips it (alternatively the one after it),
/// else returns position of the current directory
static char *skip_dir(char *start, char *end)
{
    if (end < start + 1)
        return NULL;
    if (*(end) == '/' && *(end - 1) == '.') {
        end--;
        // if '../' else './'
        if (end > start + 1 && *(end - 1) == '.') {
            end = skip_dir(start, end - 2);
            while (end - 1 > start &&
                    !(*(end - 1) == '/' && *(end - 2) != '/'))
                end--;
            end--;
        } else
            end = skip_dir(start, end - 1);
    }
    return end;
}

char *normalize_path(char *path_old)
{
    if (strchr("~/\\", path_old[0]) || strstr(path_old, ":/") || strstr(path_old, ":\\"))
        return NULL;
    size_t length = 1;
    int i = 0;
    char *path = malloc(strlen(path_old) + 2);
    if (!path)
        return NULL;
    strcpy(path, path_old);
    // gets rid of multiples of '/' or '\'
    for (unsigned x = 1; path[x] != 0; x++) {
        if (!((path[x] == '/' && path[x - 1] == '/') ||
                    (path[x] == '\\' && path[x - 1] == '\\'))) {
            path[length++] = path[x];
        }
    }
    // resolves './'
    for (unsigned x = 0; x < length; x++) {
        if (path[x] == '.' && path[x + 1] == '/' &&
                (i < 1 || path[x - 1] != '.'))
            x++;
        else
            path[i++] = path[x];
    }
    path[i] = 0, length = i;
    // resolves '../'
    for (char *x = path + length; x >= path; i--, x--) {
        if (path[i] == '/') {
            x = skip_dir(path, x);
            if (x < path || x >= path + i + 1) {
                free(path);
                return NULL;
            }
            if (x == path)
                break;
        }
        path[i] = *x;
    }
    char *normalized = malloc(length - i + 1);
    strcpy(normalized, path + i + 1);
    free(path);
    return normalized;
}

char *create_prefix(const char *path)
{
    struct view prefix_view = view_create(path, -1);
    view_tail(&prefix_view, '.');
    prefix_view.end++;
    char *prefix = view_materialize(prefix_view);
    prefix[view_length(prefix_view) - 1] = ':';
    for (int i = 0; i < view_length(prefix_view); i++) {
        if (prefix[i] == '/')
            prefix[i] = ':';
        else if (!valid_chr(prefix[i]))
            prefix[i] = '?';
    }
    return prefix;
}
