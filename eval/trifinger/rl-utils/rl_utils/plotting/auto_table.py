from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from rlf.exp_mgr.wb_query import query

MISSING_VALUE = 0.2444


def get_df_for_table_txt(
    table_txt: str, lookup_k: str
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Extracts a dataframe to use for `plot_table` automatically from text copied
    from excel. You want to include the row and column names in the text.
    Example:
    ```
            mirl train	mirl eval	airl train	airl eval
    100	5TK1	UKGZ	YIGE	GN31
    50	14MT	C0JW	KUOP	OVS2
    ```
    If you are getting eval metrics, `lookup_k` should likely be 'final_train_success'.
    """
    data = []
    row_headers = []
    for line in table_txt.split("\n"):
        if line.strip() == "":
            continue
        line_parts = line.split("\t")
        row_ident = line_parts[0].strip()
        if row_ident == "":
            # These are the column headers.
            col_headers = line_parts[1:]
        else:
            assert len(line_parts[1:]) == len(col_headers)
            row_headers.append(row_ident)
            for group, col in zip(line_parts[1:], col_headers):
                r = query([lookup_k], {"group": group}, use_cached=True)
                if r is None or len(r) == 0:
                    r = [{lookup_k: MISSING_VALUE}]

                for d in r:
                    data.append(
                        {"method": row_ident, "type": col, lookup_k: d[lookup_k]}
                    )
    return pd.DataFrame(data), col_headers, row_headers


def def_make_col_header(n_cols):
    return "c" * n_cols


def plot_table(
    df: pd.DataFrame,
    col_key: str,
    row_key: str,
    cell_key: str,
    col_order: List[str],
    row_order: List[str],
    renames: Optional[Dict[str, str]] = None,
    error_scaling=1.0,
    n_decimals=2,
    missing_fill_value=MISSING_VALUE,
    error_fill_value=0.3444,
    get_row_highlight: Optional[Callable[[str, pd.DataFrame], Optional[str]]] = None,
    make_col_header: Callable[[int], str] = def_make_col_header,
    x_label: str = "",
    y_label: str = "",
    skip_toprule: bool = False,
    write_to=None,
):
    """
    :param df: The index of the data frame does not matter, only the row values and column names matter.
    :param col_key: A string from the set of columns.
    :param row_key: A string from the set of columns (but this is used to form the rows of the table).
    :param renames: Only used for display name conversions. Does not affect functionality.
        Example: the data fame might look like
        ```
           democount        type  final_train_success
        0     100  mirl train               0.9800
        1     100  mirl train               0.9900
        3     100   mirl eval               1.0000
        4     100   mirl eval               1.0000
        12     50  mirl train               0.9700
        13     50  mirl train               1.0000
        15     50   mirl eval               1.0000
        16     50   mirl eval               0.7200
        ```
        `col_key='type', row_key='demcount',
        cell_key='final_train_success'` plots the # of demos as rows and
        the type as columns with the final_train_success values as the cell
        values. Duplicate row and columns are automatically grouped
        together.
    :param make_col_header: Returns the string at the top of the table like
        "ccccc". Put "c|ccccc" to insert a vertical line in between the first
        and other columns.
    :param x_label: Renders another row of text on the top that spans all the columns.
    ;param y_label: Renders a side column with vertically rotated text that spawns all the rows.
    """

    if renames is None:
        renames = {}
    df = df.replace("missing", missing_fill_value)
    df = df.replace("error", error_fill_value)

    rows = {}
    for row_k, row_df in df.groupby(row_key):
        df_avg_y = row_df.groupby(col_key)[cell_key].mean()
        df_std_y = row_df.groupby(col_key)[cell_key].std() * error_scaling

        rows[row_k] = (df_avg_y, df_std_y)

    col_sep = " & "
    row_sep = " \\\\\n"

    all_s = []

    def clean_text(s):
        return s.replace("%", "\\%").replace("_", " ")

    # Add the column title row.
    row_str = [""]
    for col_k in col_order:
        row_str.append("\\textbf{%s}" % clean_text(renames.get(col_k, col_k)))
    all_s.append(col_sep.join(row_str))

    for row_k in row_order:
        if row_k == "hline":
            all_s.append("\\hline")
            continue
        row_str = []
        row_str.append("\\textbf{%s}" % clean_text(renames.get(row_k, row_k)))
        row_y, row_std = rows[row_k]

        if get_row_highlight is not None:
            sel_col = get_row_highlight(row_k, row_y)
        else:
            sel_col = None
        for col_k in col_order:
            if col_k not in row_y:
                row_str.append("-")
            else:
                val = row_y.loc[col_k]
                std = row_std.loc[col_k]
                if val == missing_fill_value:
                    row_str.append("-")
                elif val == error_fill_value:
                    row_str.append("E")
                else:
                    if col_k == sel_col:
                        row_str.append(
                            "\\textbf{ "
                            + (
                                f"%.{n_decimals}f {{\\scriptsize $\\pm$ %.{n_decimals}f }}"
                                % (val, std)
                            )
                            + " }"
                        )
                    else:
                        row_str.append(
                            f" %.{n_decimals}f {{\\scriptsize $\\pm$ %.{n_decimals}f }} "
                            % (val, std)
                        )

        all_s.append(col_sep.join(row_str))

    n_columns = len(col_order) + 1
    col_header_s = make_col_header(n_columns)
    if y_label != "":
        col_header_s = "c" + col_header_s
        start_of_line = " & "
        toprule = ""

        midrule = "\\cmidrule{2-%s}\n" % (n_columns + 1)
        botrule = midrule
        row_lines = [start_of_line + x for x in all_s[1:]]
        row_lines[0] = (
            "\\multirow{4}{1em}{\\rotatebox{90}{%s}}" % y_label
        ) + row_lines[0]
    else:
        row_lines = all_s[1:]
        start_of_line = ""
        toprule = "\\toprule\n"
        midrule = "\\midrule\n"
        botrule = "\\bottomrule"

    if skip_toprule:
        toprule = ""

    if x_label != "":
        toprule += ("& \\multicolumn{%i}{c}{%s}" % (n_columns, x_label)) + row_sep

    ret_s = ""
    ret_s += "\\begin{tabular}{%s}\n" % col_header_s
    # Line above the table.
    ret_s += toprule

    # Separate the column headers from the rest of the table by a line.
    ret_s += start_of_line + all_s[0] + row_sep
    ret_s += midrule

    all_row_s = ""
    for row_line in row_lines:
        all_row_s += row_line
        if "hline" not in row_line:
            all_row_s += row_sep
        else:
            all_row_s += "\n"

    ret_s += all_row_s
    # Line below the table.
    ret_s += botrule

    ret_s += "\n\\end{tabular}\n"

    if write_to is not None:
        with open(write_to, "w") as f:
            f.write(ret_s)
        print(f"Wrote result to {write_to}")
    else:
        print(ret_s)

    return ret_s
