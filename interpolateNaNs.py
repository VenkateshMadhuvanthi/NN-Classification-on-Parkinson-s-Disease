# -*- coding: utf-8 -*-
import pandas as pd
#%%

def interpolateNaNs(inputDf,
                    allowNaNRows=True,
                    allowNaNCols=False,
                    dropEmptyEntries=False,
                    handleNoMatch='cleanMean'):
    """
    Takes a dataframe and interpolates missing entries from the mean of all
    rows that have the same entries in the non-NaN columns. Returns the
    interpolated dataframe.

    Inputs:
        * inputDf : dataframe to be interpolated
        * allowNaNCols : bool that decides whether to abort interpolation when
                        an entire column is Nan.
        * allowNaNRows : bool that decides whether to abort interpolation when
                        an entire row is Nan.
        * dropEmptyEntries : bool that decides whether to drop completely empty
                            rows
        * handleNoMatch : Decides how to fill in values for rows that don't
                        match any other rows (including full-NaN rows)
                        Currently implemented:
                            * 'pass' : do nothing
                            * 'totalMean' : take mean from entire inputDf,
                                skipping NaNs
                            * 'cleanMean' : take mean from non-empty rows
    """
    if inputDf.isna().all(axis=0).any() and not allowNaNCols:
        raise ValueError("At least one entire column is NaNs. \
                         Set allowNaNCols=True to ignore.")
    if (inputDf.isna().all(axis=1).any()
            and not allowNaNRows
            and not dropEmptyEntries):
        raise ValueError("At least one entire row is NaNs. \
                         Set allowNaNRows=True to ignore or \
                         dropEmptyEntries=True to remove empty rows.")

    # copy the input df (probably not necessary but safe)
    interpolateDf = inputDf.copy()
    # drop empty rows if present
    if dropEmptyEntries:
        interpolateDf = interpolateDf[~interpolateDf.isna().all(axis=1)]
    # mask all NaN row indices
    nanmask = ~inputDf.isna().any(axis=1)
    # separate input df into a "clean" df and a df with NaNs
    cleanDf = inputDf[nanmask]
    nanDf = inputDf[~nanmask]
    # iterate through all NaN entries
    for idx, row in nanDf.iterrows():

        # find columns that don't contain NaNs
        naNColumns = nanDf.columns[row.isna()]
        nonNaNColumns = nanDf.columns[~row.isna()]
        # take part of row that does not contain NaNs
        nonNaNPart = row[nonNaNColumns]
        match = (cleanDf[nonNaNColumns] == nonNaNPart)

        # decide how to handle rows without match in cleanDf or all-NaN rows
        if match.empty or row.isna().all():
            newRow = row.copy()
            if handleNoMatch == 'pass':
                # just ignore entry keeping it NaN
                continue
            elif handleNoMatch == 'totalMean':
                # calculate mean of all entries
                newRow[naNColumns] = inputDf[naNColumns].mean(axis=0,
                                                              skipna=True)
            elif handleNoMatch == 'cleanMean':
                # calculate mean of all clean entries
                newRow[naNColumns] = cleanDf[naNColumns].mean(axis=0)
            else:
                raise ValueError(f"Unknown mode {handleNoMatch}")
        else:
            # find all entries in the clean df that match the non-NaN entries
            # of the row
            matchInCleanDf = cleanDf[match.all(axis=1)]
            newRow = matchInCleanDf.mean(axis=0, skipna=True)
        # set the interpolated df entry to the mean of of the previously
        # defined entries
        interpolateDf.loc[idx] = newRow
    return interpolateDf
#%%

if __name__ == "__main__":
    # for testing, replace this line with the path to the datalabels file
    dataLabelDf = pd.read_csv('CIS-PD_Training_Data_IDs_Labels.csv').set_index(['subject_id',
                                                   'measurement_id'])
    interpolateDf = interpolateNaNs(dataLabelDf)
