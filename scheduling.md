# Scheduling and Logs

## Running on a schedule

Fluxo has no built-in scheduler. Use cron or a workflow tool to call `fluxo sync` at the interval you need.

## Retries

Failed jobs are retried up to 3 times with a delay that doubles after each attempt, starting at 2 seconds. Set --retries 0 to disable retries.

## Log files

Logs are written to the logs folder in the project directory. Each run creates a file named by its start time, and files older than 14 days are deleted automatically.

## Log verbosity

Pass --verbose to include per-table row counts in the log. The default level shows only the summary and errors.
