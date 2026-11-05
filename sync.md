# Running Sync Jobs

## Basic sync

Use `fluxo sync --source prod --target warehouse` to copy tables from the source connection to the target connection.

## Dry runs

Add the --dry-run flag to preview changes. A dry run prints the rows that would be written but never modifies the target database.

## Batch size

By default Fluxo copies rows in batches of 5000. Use --batch-size to change this. Larger batches are faster but use more memory.

## Incremental mode

With --incremental, Fluxo copies only rows whose updated_at column is newer than the last successful run. The last run time is stored in a file named .fluxo_state in the project directory.
