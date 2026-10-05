# Data

The raw inputs are Bloomberg Terminal exports. Bloomberg data are licensed and
cannot be redistributed, so **this folder is empty in the repository** (its
contents are git-ignored). To re-run the pipeline, place the four files below
here, or set `OATCURVE_DATA_DIR` to the folder that holds them.

| File | Content | Used by |
|---|---|---|
| `Analisi OATs 1.csv` | French security master | `oatcurve.data_loading` |
| `Analisi OATs (Bid-Ask Data).csv` | Daily OAT/BTAN ask and bid prices, 22 Oct 1987 – 5 Jun 2026 | `oatcurve.data_loading` |
| `Analisi Bunds 1.csv` | German security master | `studies.bund_curve` |
| `Analisi Bunds (Bid-Ask Data).csv` | Daily Schatz/Bobl/Bund ask and bid prices, 1 Jan 1999 – 5 Jun 2026 | `studies.bund_curve` |

The file names are those of the original exports, made from an Italian-language
Bloomberg terminal; the column headers are therefore in Italian.

## Security master (`* 1.csv`)

One row per bond. Columns read by the code (any order; extra columns ignored):

| Column | Meaning | Example |
|---|---|---|
| `Nome emittente` | Issuer name | `French Republic Government Bond OAT` |
| `ISIN` | ISIN | `FR001400QMF9` |
| `Cedola` | Annual coupon, % of face | `3` |
| `Data emissione` | Issue date, `dd/mm/yyyy` | `10/06/2024` |
| `Scadenza` | Maturity date, `dd/mm/yyyy` | `25/11/2034` |
| `Tipo scad` | Maturity type (`AT MATURITY`, `CALL/SINK`, `PUTABLE`) | `AT MATURITY` |
| `Serie` | Bloomberg series | `OAT` |
| `Valuta` | Currency at issuance (`EUR`, `FRF`, `XEU`) | `EUR` |

## Price panel (`* (Bid-Ask Data).csv`)

A wide table: the first three columns are `BONDS, PRICES, Dates`, followed by
one column per trading day (header `dd/mm/yyyy`). Each bond has two rows:

```
BONDS,PRICES,Dates,22/10/1987,23/10/1987,...
FR001400QMF9 Govt,Ask Price,PX_ASK,#N/A N/A,#N/A N/A,...
,Bid Price,PX_BID,#N/A N/A,#N/A N/A,...
```

The ISIN appears only on the ask row and is forward-filled; only the **bid**
rows are used, as in Gürkaynak, Sack & Wright (2007). Bloomberg `#N/A` tokens
become missing values. Prices are clean, per 100 of face.

## ECB series

Euro-area curves, the CISS stress index, the Eurosystem balance sheet, the
3-month Euribor and MFI lending rates are downloaded from the
[ECB Data Portal](https://data.ecb.europa.eu/) on first use and cached in
`output/studies_cache/`. Behind a TLS-intercepting proxy or antivirus, set
`ECB_SSL_VERIFY=0`.
