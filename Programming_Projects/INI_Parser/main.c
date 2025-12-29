#include "ini_parser.h"
#include "path.h"

bool ini_file(switch_t, filesystem_t *, char *filepath, list_t);


int help(FILE *output)
{
    const char help[] =
            "Transforms a .ini filesystem into a single file\n"
            "The program allows the following functionalities:\n\n"
            "\t-h, --help\n"
            "\t\tprints out this\n\n"
            "\t-d N, --max-depth N\n"
            "\t\tmaximum recursion depth, allows only numbers as args,\n "
            "\t\t0 means inclusion is excluded, negatives allow unlimited depth\n\n"
            "\t-g, --include-guard\n"
            "\t\tevery file will be inserted only once, will report cycles\n\n"
            "\t-r, --report-cycles\n"
            "\t\tprogram will report cycles, raises errors, still overrides --include-guard\n\n";
    fprintf(output, help);
    return 1;
}

bool insert_comment(section_t *section, char *line)
{
    size_t keys = section->n_of_keys;
    if (keys + 1 >= section->allocd) {
        section->keys = realloc(section->keys, sizeof(struct content_t) * (section->allocd += 5));
    }
    section->keys[keys] = (struct content_t){ NULL, line };
    section->n_of_keys++;
    return true;
}

/// retrieves key and value and checks whether the key isn't a duplicate
bool insert_key(section_t *section, char *line)
{
    size_t keys = section->n_of_keys;
    if (keys + 1 >= section->allocd) {
        section->keys = realloc(section->keys, sizeof(struct content_t) * (section->allocd += 5));
    }
    struct content_t new = key_from_input(line);
    section->keys[keys] = new;
    free(line);
    if (!section->keys[keys].key)
        return false;
    section->n_of_keys++;
    for (unsigned i = 0; i < keys; i++) {
        if (section->keys[i].key && !strcmp(new.key, section->keys[i].key)) {
            fprintf(stderr, "Duplicate key: %s\n", new.key);
            return false;
        }
    }
    return true;
}

char *get_title(char *line, char *source)
{
    char *title_wo_prefix = title_from_input(line);
    char *prefix = NULL;
    if (title_wo_prefix) {
        prefix = (char *) malloc(strlen(title_wo_prefix) + strlen(source) + 1);
        prefix = strcat(strcpy(prefix, source), title_wo_prefix);
        free(title_wo_prefix);
    }
    return prefix;
}

///@note try to find a cleverer way to trim down this function
bool parse(switch_t switches, filesystem_t *files, char *filepath, FILE *file, char *source, list_t sources)
{
    char *line = get_str(file);
    enum content_result res = get_content(line);
    files->sections[files->sec_n] = (section_t){ NULL, NULL, 0, 0 };

    for (files->sec_n++; res != end; files->sec_n++) {
        if (files->sec_n + 2 >= files->allocd)
            files->sections = (section_t *) realloc(files->sections,
                    sizeof(section_t) * (files->allocd <<= 1));
        // skips lines, until it doesn't (handles pre-section-title content)
        while (res == comment || res == include) {
            if (res == include) {
                char *next_file = create_filepath(filepath, include_from_input(line));
                bool parse_res = ini_file(switches, files, next_file, sources);
                free(next_file);
                free(line);
                if (!parse_res)
                    return false;
            } else
                insert_comment(files->sections + files->sec_n - 1, line);
            line = get_str(file);
            res = get_content(line);
        }
        files->sections[files->sec_n] = (section_t){ NULL, NULL, 0, 0 };
        if (res != end && !(files->sections[files->sec_n].title = get_title(line, source)))
            return false;
        // inserts key + values into the current section (handles post-section-title content)
        while ((res = get_content((line = get_str(file)))) == comment || res == content) {
            if (res == content && !insert_key(files->sections + files->sec_n, line)) {
                files->sec_n++;
                return false;
            }
            if (res == comment)
                insert_comment(files->sections + files->sec_n, line);
        }
    }
    return true;
}

bool ini_file(switch_t switches, filesystem_t *files, char *filepath, list_t sources)
{
    if (!filepath)
        return false;
    char *dir_path = normalize_path(filepath);
    if (!dir_path) {
        fprintf(stderr, "%s is outside of the root directory\n", filepath);
        return false;
    }
    sources.list = strcpy(malloc(sources.len + 1), sources.list);
    if (switches.strict || switches.guard) {
        enum cycle_t insert = dplct_file(switches, files, dir_path, &sources);
        if (insert != new) {
            free(sources.list);
            return !(insert == cycle && switches.strict);
        }
    } else
        insert_file(files, dir_path);
    if (switches.depth == 0) {
        free(sources.list);
        fprintf(stderr, "Recursion depth  limit reached in %s\n", filepath);
        return false;
    }
    switches.depth--;
    char *complete_path = malloc(strlen(filepath) + strlen(files->root) + 1);
    strcat(strcpy(complete_path, files->root), filepath);
    FILE *file = fopen(complete_path, "r");
    free(complete_path);
    if (!file) {
        fprintf(stderr, "%s doesn't exist\n", dir_path);
        free(sources.list);
        return false;
    }
    char *prefix = create_prefix(dir_path);
    bool correct_parse = parse(switches, files, dir_path, file, prefix, sources);
    free(sources.list);
    free(prefix);
    fclose(file);
    return correct_parse;
}

bool output(filesystem_t *files, FILE *out, bool comments)
{
    for (unsigned i = 0; i < files->sec_n; i++) {
        if (files->sections[i].title)
            fprintf(out, "[%s]\n", files->sections[i].title);
        for (unsigned x = 0; x < files->sections[i].n_of_keys; x++) {
            if (files->sections[i].keys[x].key)
                fprintf(out, "%s = ", files->sections[i].keys[x].key);
            if (files->sections[i].keys[x].key || comments)
                fprintf(out, "%s\n", files->sections[i].keys[x].value);
        }
    }
    return true;
}

bool dplct_sections(filesystem_t *files)
{
    for (unsigned i = 0; i < files->sec_n; i++) {
        if (files->sections[i].title)
            for (unsigned x = i + 1; x < files->sec_n; x++) {
                if (files->sections[x].title &&
                        !strcmp(files->sections[x].title, files->sections[i].title))
                    return true;
            }
    }
    return false;
}

int pre_parse(switch_t switches, char *input_path, char *output_path, bool comments)
{
    FILE *out = NULL, *input = fopen(input_path, "r");
    if (!input) {
        fprintf(stderr, "Source file invalid");
        return 1;
    }
    struct view root_file = view_create(input_path, -1);
    char *file = view_materialize(view_tail(&root_file, '/'));
    if (root_file.end != root_file.begin)
        root_file.end++;
    char *root = view_materialize(root_file), **files = malloc(2 * sizeof(void *));
    files[0] = file;
    list_t sources = { NULL, strlen(file) + 1 };
    char *first_file = malloc(sources.len + 1);
    strcpy(first_file, file);
    first_file[sources.len - 1] = ' ';
    first_file[sources.len] = 0;
    sources.list = first_file;

    filesystem_t filesystem = { files, 1, root, malloc(sizeof(section_t) * 5), 0, 5 };
    if (parse(switches, &filesystem, file, input, "", sources)) {
        if (!dplct_sections(&filesystem)) {
            out = fopen(output_path, "w");
            if (out) {
                output(&filesystem, out, comments);
                fclose(out); // not ideal :/
            }
        } else
            fprintf(stderr, "Duplicitous section\n");
    }
    free(sources.list);
    fclose(input);
    free_filesystem(&filesystem);
    return out == NULL;
}

int main(int argc, char **argv)
{
    switch_t switches = { false, false, 10 };
    bool comments = false;
    for (int i = 1; i < argc - 2; i++) {
        const char *option = argv[i];
        if (!strcmp("-d", option) || !strcmp("--max-depth", option)) {
            if (++i < argc && isdigit(argv[i][0])) {
                int depth = atoi(argv[i]);
                switches.depth = depth >= 0 ? depth : -1;
            } else
                return help(stderr);
        } else if (!strcmp("-g", option) || !strcmp("--include-guard", option)) {
            switches.guard = true;
        } else if (!strcmp("-r", option) || !strcmp("--report-cycles", option)) {
            switches.strict = true;
        } else if (!strcmp("-h", option) || !strcmp("--help", option)) {
            help(stdout);
        } else if (!strcmp("-c", option) || !strcmp("--with-comments", option)) {
            comments = true;
        } else {
            help(stderr);
            return 1;
        }
    }
    return pre_parse(switches, argv[argc - 2], argv[argc - 1], comments);
}
