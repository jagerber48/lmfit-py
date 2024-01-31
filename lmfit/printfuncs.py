"""Functions to display fitting results and confidence intervals."""

from math import log10
import re

import numpy as np
from sciform import Formatter
from tabulate import tabulate

try:
    import numdifftools  # noqa: F401
    HAS_NUMDIFFTOOLS = True
except ImportError:
    HAS_NUMDIFFTOOLS = False


def alphanumeric_sort(s, _nsre=re.compile('([0-9]+)')):
    """Sort alphanumeric string."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


def getfloat_attr(obj, attr, length=11):
    """Format an attribute of an object for printing."""
    float_formatter = Formatter(exp_mode="fixed_point", ndigits=4)
    val = getattr(obj, attr, None)
    if val is None:
        return 'unknown'
    if isinstance(val, int):
        return f'{val}'
    if isinstance(val, float):
        return float_formatter(val)
    return repr(val)


def gformat(val, length=11):
    """Format a number with '%g'-like format.

    Except that:
        a) the length of the output string will be of the requested length.
        b) positive numbers will have a leading blank.
        b) the precision will be as high as possible.
        c) trailing zeros will not be trimmed.

    The precision will typically be ``length-7``.

    Parameters
    ----------
    val : float
        Value to be formatted.
    length : int, optional
        Length of output string (default is 11).

    Returns
    -------
    str
        String of specified length.

    Notes
    ------
    Positive values will have leading blank.

    """
    if val is None or isinstance(val, bool):
        return f'{repr(val):>{length}s}'
    try:
        expon = int(log10(abs(val)))
    except (OverflowError, ValueError):
        expon = 0
    except TypeError:
        return f'{repr(val):>{length}s}'

    length = max(length, 7)
    form = 'e'
    prec = length - 7
    if abs(expon) > 99:
        prec -= 1
    elif ((expon > 0 and expon < (prec+4)) or
          (expon <= 0 and -expon < (prec-1))):
        form = 'f'
        prec += 4
        if expon > 0:
            prec -= expon
    return f'{val:{length}.{prec}{form}}'


def indent_string_block(string_block, depth=4):
    indent = " "*depth
    string_block = f"{indent}{string_block}"
    string_block = string_block.replace("\n", f"\n{indent}")
    return string_block


def fit_report(inpars, modelpars=None, show_correl=True, min_correl=0.1,
               sort_pars=False, correl_mode='list'):
    """Generate a report of the fitting results.

    The report contains the best-fit values for the parameters and their
    uncertainties and correlations.

    Parameters
    ----------
    inpars : Parameters
        Input Parameters from fit or MinimizerResult returned from a fit.
    modelpars : Parameters, optional
        Known Model Parameters.
    show_correl : bool, optional
        Whether to show list of sorted correlations (default is True).
    min_correl : float, optional
        Smallest correlation in absolute value to show (default is 0.1).
    sort_pars : bool or callable, optional
        Whether to show parameter names sorted in alphanumerical order. If
        False (default), then the parameters will be listed in the order
        they were added to the Parameters dictionary. If callable, then
        this (one argument) function is used to extract a comparison key
        from each list element.
    correl_mode : {'list', table'} str, optional
        Mode for how to show correlations. Can be either 'list' (default)
        to show a sorted (if ``sort_pars`` is True) list of correlation
        values, or 'table' to show a complete, formatted table of
        correlations.

    Returns
    -------
    str
        Multi-line text of fit report.

    """
    from .parameter import Parameters
    if isinstance(inpars, Parameters):
        result, params = None, inpars
    if hasattr(inpars, 'params'):
        result = inpars
        params = inpars.params

    if sort_pars:
        if callable(sort_pars):
            key = sort_pars
        else:
            key = alphanumeric_sort
        parnames = sorted(params, key=key)
    else:
        # dict.keys() returns a KeysView in py3, and they're indexed
        # further down
        parnames = list(params.keys())

    buff = []
    add = buff.append
    namelen = max(len(n) for n in parnames)
    if result is not None:
        add("[[Fit Statistics]]")
        fit_stats_data = [
            ["# fitting method", result.method],
            ["# function evals", getfloat_attr(result, 'nfev')],
            ["# data points", getfloat_attr(result, 'ndata')],
            ["# variables", getfloat_attr(result, 'nvarys')],
            ["chi-square", getfloat_attr(result, 'chisqr')],
            ["reduced chi-square", getfloat_attr(result, 'redchi')],
            ["Akaike info crit", getfloat_attr(result, 'aic')],
            ["Bayesian info crit", getfloat_attr(result, 'bic')],
        ]
        if hasattr(result, 'rsquared'):
            fit_stats_data.append(["R-squared", getfloat_attr(result, 'rsquared')])

        fit_stats_table = tabulate(fit_stats_data, disable_numparse=True)
        fit_stats_table = indent_string_block(fit_stats_table)
        add(fit_stats_table)

        if not result.errorbars:
            add("##  Warning: uncertainties could not be estimated:")
            if result.method in ('leastsq', 'least_squares') or HAS_NUMDIFFTOOLS:
                parnames_varying = [par for par in result.params
                                    if result.params[par].vary]
                for name in parnames_varying:
                    par = params[name]
                    space = ' '*(namelen-len(name))
                    if par.init_value and np.allclose(par.value, par.init_value):
                        add(f'    {name}:{space}  at initial value')
                    if (np.allclose(par.value, par.min) or np.allclose(par.value, par.max)):
                        add(f'    {name}:{space}  at boundary')
            else:
                add("    this fitting method does not natively calculate uncertainties")
                add("    and numdifftools is not installed for lmfit to do this. Use")
                add("    `pip install numdifftools` for lmfit to estimate uncertainties")
                add("    with this fitting method.")

    val_unc_formatter = Formatter(exp_mode="fixed_pointfit", ndigits=2, superscript=True)
    var_data = []
    add("[[Variables]]")
    for name in parnames:
        single_var_data = {
            "Name": None,
            "Value": None,
            "Percent Uncertainty": None,
            "Constraint": None,
            "Init Val": None,
            "Model Val": None,
        }
        par = params[name]
        single_var_data["Name"] = name

        if par.init_value is not None:
            single_var_data["Init Val"] = f"{par.init_value:.7g}"
        else:
            single_var_data["Init Val"] = ""

        if modelpars is not None and name in modelpars:
            single_var_data["Model Val"] = f"{modelpars[name].value:.7g}"
        else:
            single_var_data["Model Val"] = ""

        val = par.value
        if not isinstance(val, (int, float)):
            single_var_data["Value"] = "Non Numeric Value?"
        else:
            stderr = par.stderr
            if stderr is not None:
                single_var_data["Value"] = val_unc_formatter(val, stderr)
                try:
                    single_var_data[
                        "Percent Uncertainty"] = f'{abs(par.stderr / par.value):.2%}'
                except ZeroDivisionError:
                    single_var_data["Percent Uncertainty"] = ""
            else:
                single_var_data["Value"] = f"{val:.7g}"
                single_var_data["Percent Uncertainty"] = ""

        if par.vary:
            single_var_data["Constraint"] = "Vary"
        elif par.expr is not None:
            single_var_data["Constraint"] = par.expr
        else:
            single_var_data["Constraint"] = "Fixed"

        var_data.append(single_var_data)

    var_table = tabulate(var_data, headers="keys", tablefmt="simple", numalign="center",
                         stralign="center", disable_numparse=True)
    var_table = indent_string_block(var_table)
    add(var_table)

    if show_correl and correl_mode.startswith('tab'):
        add('[[Correlations]] ')
        correl_table_str = correl_table(params)
        correl_table_str = indent_string_block(correl_table_str)
        add(correl_table_str)
    elif show_correl:
        correls = {}
        for i, name in enumerate(parnames):
            par = params[name]
            if not par.vary:
                continue
            if hasattr(par, 'correl') and par.correl is not None:
                for name2 in parnames[i+1:]:
                    if (name != name2 and name2 in par.correl and
                            abs(par.correl[name2]) > min_correl):
                        correls[f"{name}, {name2}"] = par.correl[name2]

        sort_correl = sorted(correls.items(), key=lambda it: abs(it[1]))
        sort_correl.reverse()
        if len(sort_correl) > 0:
            add('[[Correlations]] (unreported correlations are < '
                f'{min_correl:.3f})')
            maxlen = max(len(k) for k in list(correls.keys()))
        for name, val in sort_correl:
            lspace = max(0, maxlen - len(name))
            add(f"    C({name}){(' '*30)[:lspace]} = {val:+.4f}")
    return '\n'.join(buff)


def lcol(s, cat='td'):
    "html left column"
    return f"<{cat} style='text-align:left'>{s}</{cat}>"


def rcol(s, cat='td'):
    "html right column"
    return f"<{cat} style='text-align:right'>{s}</{cat}>"


def trow(columns, cat='td'):
    "html row"
    nlast = len(columns)-1
    rows = []
    for i, col in enumerate(columns):
        cform = rcol if i == nlast else lcol
        rows.append(cform(col, cat=cat))
    return rows


def fitreport_html_table(result, show_correl=True, min_correl=0.1):
    """Generate a report of the fitting result as an HTML table.

    Parameters
    ----------
    result : MinimizerResult or ModelResult
        Object containing the optimized parameters and several
        goodness-of-fit statistics.
    show_correl : bool, optional
        Whether to show list of sorted correlations (default is True).
    min_correl : float, optional
        Smallest correlation in absolute value to show (default is 0.1).

    Returns
    -------
    str
        Multi-line HTML code of fit report.

    """
    html = []
    add = html.append

    def stat_row(label, val, val2=None, cat='td'):
        if val2 is None:
            rows = trow([label, val], cat=cat)
        else:
            rows = trow([label, val, val2], cat=cat)
        add(f"<tr>{''.join(rows)}</tr>")

    add('<table class="jp-toc-ignore">')
    add('<caption class="jp-toc-ignore">Fit Statistics</caption>')
    stat_row('fitting method', result.method)
    stat_row('# function evals', result.nfev)
    stat_row('# data points', result.ndata)
    stat_row('# variables', result.nvarys)
    stat_row('chi-square', gformat(result.chisqr))
    stat_row('reduced chi-square', gformat(result.redchi))
    stat_row('Akaike info crit.', gformat(result.aic))
    stat_row('Bayesian info crit.', gformat(result.bic))
    if hasattr(result, 'rsquared'):
        stat_row('R-squared', gformat(result.rsquared))
    add('</table>')
    add(params_html_table(result.params))
    if show_correl:
        correls = []
        parnames = list(result.params.keys())
        for i, name in enumerate(result.params):
            par = result.params[name]
            if not par.vary:
                continue
            if hasattr(par, 'correl') and par.correl is not None:
                for name2 in parnames[i+1:]:
                    if (name != name2 and name2 in par.correl and
                            abs(par.correl[name2]) > min_correl):
                        correls.append((name, name2, par.correl[name2]))
        if len(correls) > 0:
            sort_correls = sorted(correls, key=lambda val: abs(val[2]))
            sort_correls.reverse()
            extra = f'(unreported values are < {min_correl:.3f})'
            add('<table class="jp-toc-ignore">')
            add(f'<caption>Correlations {extra}</caption>')
            stat_row('Parameter1', 'Parameter 2', 'Correlation', cat='th')
            for name1, name2, val in sort_correls:
                stat_row(name1, name2, f"{val:+.4f}")
            add('</table>')
    return ''.join(html)


def correl_table(params):
    varnames = [vname for vname in params if params[vname].vary]

    correl_data = []

    for vname in varnames:
        var_data_dict = dict.fromkeys([""] + varnames)
        var_data_dict[""] = vname
        par = params[vname]
        for vother in varnames:

            if vother == vname:
                var_data_dict[vother] = f"{1.0:+.4f}"
            elif vother in par.correl:
                var_data_dict[vother] = f"{par.correl[vother]:+.4f}"
            else:
                var_data_dict[vother] = "unknown"
        correl_data.append(var_data_dict)

    correl_table_str = tabulate(correl_data, headers="keys", tablefmt="simple_grid",
                                disable_numparse=True)
    return correl_table_str


def params_html_table(params):
    """Return an HTML representation of Parameters.

    Parameters
    ----------
    params : Parameters
        Object containing the Parameters of the model.

    Returns
    -------
    str
        Multi-line HTML code of fitting parameters.

    """
    has_err = any(p.stderr is not None for p in params.values())
    has_expr = any(p.expr is not None for p in params.values())
    has_brute = any(p.brute_step is not None for p in params.values())

    html = []
    add = html.append

    add('<table class="jp-toc-ignore"><caption>Parameters</caption>')
    headers = ['name', 'value']
    if has_err:
        headers.extend(['standard error', 'relative error'])
    headers.extend(['initial value', 'min', 'max', 'vary'])
    if has_expr:
        headers.append('expression')
    if has_brute:
        headers.append('brute step')

    hrow = trow(headers, cat='th')
    add(f"<tr>{''.join(hrow)}</tr>")

    for par in params.values():
        rows = [par.name, gformat(par.value)]
        if has_err:
            serr = ''
            spercent = ''
            if par.stderr is not None:
                serr = gformat(par.stderr)
                try:
                    spercent = f'({abs(par.stderr/par.value):.2%})'
                except ZeroDivisionError:
                    pass
            rows.extend([serr, spercent])
        rows.extend((par.init_value, gformat(par.min),
                     gformat(par.max), f'{par.vary}'))
        if has_expr:
            expr = ''
            if par.expr is not None:
                expr = par.expr
            rows.append(expr)
        if has_brute:
            brute_step = 'None'
            if par.brute_step is not None:
                brute_step = gformat(par.brute_step)
            rows.append(brute_step)

        hrow = trow(rows, cat='td')
        add(f"<tr>{''.join(hrow)}</tr>")
    add('</table>')
    return ''.join(html)


def report_fit(params, **kws):
    """Print a report of the fitting results."""
    print(fit_report(params, **kws))


def ci_report(ci, with_offset=True, ndigits=5):
    """Return text of a report for confidence intervals.

    Parameters
    ----------
    ci : dict
        The result of :func:`~lmfit.confidence.conf_interval`: a dictionary
        containing a list of ``(sigma, vals)``-tuples for each parameter.
    with_offset : bool, optional
        Whether to subtract best value from all other values (default is
        True).
    ndigits : int, optional
        Number of significant digits to show (default is 5).

    Returns
    -------
    str
        Text of formatted report on confidence intervals.

    """
    maxlen = max(len(i) for i in ci)
    buff = []
    add = buff.append

    def convp(x):
        """Convert probabilities into header for CI report."""
        if abs(x[0]) < 1.e-2:
            return "_BEST_"
        return f"{x[0] * 100:.2f}%"

    title_shown = False
    fmt_best = fmt_diff = "{0:.%if}" % ndigits
    if with_offset:
        fmt_diff = "{0:+.%if}" % ndigits
    for name, row in ci.items():
        if not title_shown:
            add("".join([''.rjust(maxlen+1)] + [i.rjust(ndigits+5)
                                                for i in map(convp, row)]))
            title_shown = True
        thisrow = [f" {name.ljust(maxlen)}:"]
        offset = 0.0
        if with_offset:
            for cval, val in row:
                if abs(cval) < 1.e-2:
                    offset = val
        for cval, val in row:
            if cval < 1.e-2:
                sval = fmt_best.format(val)
            else:
                sval = fmt_diff.format(val-offset)
            thisrow.append(sval.rjust(ndigits+5))
        add("".join(thisrow))

    return '\n'.join(buff)


def report_ci(ci):
    """Print a report for confidence intervals."""
    print(ci_report(ci))
