#include "ini_parser.h"


int valid_chr(int chr)
{
    return isalnum(chr) || strchr("!-.:?_", chr);
}

enum content_result get_content(char *line)
{
    if (!line)
        return end;
    char *cont = line;
    while (*cont != 0 && isspace(*cont))
        cont++;
    if (*cont == 0 || *cont == ';')
        return comment;
    if (*cont == '.')
        return include;
    if (*cont == '[')
        return title;
    return content;
}

char *title_from_input(char *line)
{
    char *title = NULL;
    struct view view = view_create(line, -1);
    view = view_trim(view, NULL);
    view.end--;
    if (!view_empty(view) && *(view.begin++) == '[' && *(view.end) == ']') {
        view = view_trim(view, NULL);
        if (!view_empty(view) && view_all(view, &valid_chr)) {
            title = view_materialize(view);
        } else
            fprintf(stderr, "invalid section name\n");
    } else
        fprintf(stderr, "garbage before [ or after ]\n");
    free(line);
    return title;
}

struct content_t key_from_input(char *line)
{
    struct content_t key = { NULL, NULL };
    struct view tail = view_create(line, -1);
    tail = view_trim_front(tail, NULL);
    struct view key_view = view_trim_back(view_head(&tail, '='), NULL);
    if (strchr(line, '=') && !view_empty(key_view) && view_all(key_view, valid_chr)) {
        key.key = view_materialize(key_view);
        key.value = view_materialize(view_trim(tail, NULL));
    } else
        fprintf(stderr, "invalid key\n");
    return key;
}

char *include_from_input(char *line)
{
    char *include = NULL;
    struct view view = view_create(line, -1);
    view = view_trim(view, NULL);
    struct view line_include = view_head(&view, ' ');
    if (!view_cmp(line_include, view_create(".include", -1))) {
        view = view_trim_front(view, NULL);
        include = view_materialize(view);
        if (strlen(include) < 4 || strstr(include + strlen(include) - 3, ".ini")) {
            free(include);
            fprintf(stderr, ".ini not found\n");
            return NULL;
        }
    } else
        fprintf(stderr, "invalid command after \'.\'");
    return include;
}

char *get_str(FILE *file)
{
    if (feof(file))
        return NULL;
    size_t size = 10;
    char curr;
    char *line = malloc(size);
    char *line_realloc;
    unsigned i = 0;
    for (; (curr = fgetc(file)) != EOF && curr != '\n'; i++) {
        if (i == size - 1) {
            line_realloc = (char *) realloc(line, (size <<= 1));
            if (!line_realloc) {
                free(line);
                return NULL;
            }
            line = line_realloc;
        }
        line[i] = curr;
    }
    line[i] = 0;
    return line;
}

bool free_filesystem(filesystem_t *files)
{
    for (unsigned i = 0; i < files->size; i++) {
        free(files->files[i]);
    }
    free(files->files);
    for (unsigned i = 0; i < files->sec_n; i++) {
        free(files->sections[i].title);
        for (unsigned x = 0; x < files->sections[i].n_of_keys; x++) {
            free(files->sections[i].keys[x].key);
            free(files->sections[i].keys[x].value);
        }
        free(files->sections[i].keys);
    }
    free(files->sections);
    free(files->root);
    return true;
}
