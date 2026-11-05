# Error Codes

## E101 Missing configuration

E101 means Fluxo could not find a configuration file. Create fluxo.toml or set FLUXO_HOME.

## E204 Connection timeout

E204 means a database did not respond before the timeout expired. Increase timeout_seconds or check network access to the database host.

## E310 Schema mismatch

E310 means a column in the source table does not exist in the target table. Run `fluxo schema diff` to see the differences.

## E417 Permission denied

E417 means the database user lacks write permission on the target. Grant INSERT and UPDATE rights to the user.
