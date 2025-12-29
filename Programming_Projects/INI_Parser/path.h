#include "ini_parser.h"

/// checks if file is already not in files, if not,
/// allocates extra memory and adds the file into the filesystem
enum cycle_t dplct_file(switch_t switches, filesystem_t *files, char *filename, list_t *sources);

/// adds path of source to relative path of new, free's include
char *create_filepath(char *source, char *include);

/// trims down path by resolving '../' and './', also gets rid of duplicate '/' or '\'
char *normalize_path(char *path_old);

/// replaces '/' with ':', excludes '.ini' and the root directory
char *create_prefix(const char *path);

void insert_file(filesystem_t *, char *);
