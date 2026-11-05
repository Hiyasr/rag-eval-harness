# Configuration

Fluxo reads its settings from a file called fluxo.toml in the current directory.

## Config file location

If fluxo.toml is not found, Fluxo looks in the folder given by the FLUXO_HOME environment variable. If neither exists, Fluxo stops with error E101.

## Connections

Each database connection is declared in a connections section. The default connection timeout is 30 seconds, and it can be changed with the timeout_seconds key.

## Secrets

Never store passwords in fluxo.toml. Fluxo reads passwords from environment variables named FLUXO_PASSWORD_ followed by the connection name in uppercase.
