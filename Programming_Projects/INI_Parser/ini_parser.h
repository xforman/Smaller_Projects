#ifndef INI_PARSER_H
#define INI_PARSER_H
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>
#include "view.h"

/** the structure of the filesystem_t struct is as follows:
 * filesystem_t      .files[i]
 *                   .sections[x].title
 *                              .keys[z].key
 *                                      .value
 *                              .n_of_keys;
 *                    .n_sections
 *                    .size
*/
struct filesystem_t;

typedef struct switch_t
{
    bool guard;
    bool strict;
    int depth;
} switch_t;

struct content_t
{
    char *key;
    char *value;
};

typedef struct section_t
{
    char *title;
    struct content_t *keys;
    size_t n_of_keys;
    size_t allocd;
} section_t;

typedef struct list_w_len
{
    char *list;
    size_t len;
} list_t;


typedef struct filesystem_t
{
    char **files;
    size_t size;
    char *root;
    section_t *sections;
    size_t sec_n;
    size_t allocd;
} filesystem_t;

enum cycle_t
{
    cycle,
    duplicate,
    new
};

enum content_result
{
    include,
    empty,
    comment,
    title,
    content,
    end
};

/**
 * @brief creates a section title
 * @param line an untrimmed section title
 * @param section new title is stored here on success
 * @note also free's line
 * @return true on success
 */
char *title_from_input(char *line);

/// decides which value is stored in the line as per the 'content_result' enum
enum content_result get_content(char *);

/// retrieves the key and value from a given line, returns key.NULL on failure
struct content_t key_from_input(char *);

/// checks if character is valid as per the assignment
int valid_chr(int chr);

/// takes an include line and returns a the filepath on success, NULL if ".include" is invalid
char *include_from_input(char *);

/// retrieves a full line from file, returns the malloc'd line
char *get_str(FILE *file);

bool free_filesystem(struct filesystem_t *);
#endif
