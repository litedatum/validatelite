  ---

  Issue 1: Refactor CLI and Update `check` Command

  Title: feat(cli): Refactor to use --conn/--table arguments and update check command

  Description:

  This issue covers the foundational refactoring of the CLI interface. The goal is to modernize the command structure by replacing the positional <data_source>
  argument with explicit --conn and --table options. This change will improve clarity and enable future multi-table features.

  This task includes updating the check command to be fully compatible with the new interface.

  Acceptance Criteria:
   - [ ] The positional <data_source> argument is deprecated for all commands.
   - [ ] A clear deprecation warning is shown to users who invoke the tool with the old format.
   - [ ] New mandatory options, --conn (for connection string/path) and --table (for table name), are added to the check command.
   - [ ] The internal logic of the check command is updated to correctly use the values from the new options.
   - [ ] All existing tests for the check command are updated and pass with the new interface.

  ---

  Issue 2: Implement Multi-Table Validation for `schema` Command

  Title: feat(schema): Implement multi-table validation for schema command

  Description:

  This issue focuses on enhancing the schema command to support validating multiple database tables from a single data source in one run. This is a key feature
  for improving the tool's utility in complex environments.

  Depends on: Completion of Issue #1 (CLI Refactoring).

  Acceptance Criteria:
   - [ ] The schema command is updated to use the new --conn option. It should not require a --table option, as the target tables will be defined within the rules
     file.
   - [ ] The command can successfully parse a new multi-table --rules file format, where the top-level JSON is an object with table names as keys.
   - [ ] The CLI's rule decomposition logic is updated to iterate through each table defined in the rules file and generate a complete list of atomic rules for the
     core engine.
   - [ ] The command's output is clearly grouped by table name to make results easy to interpret.
   - [ ] New unit and integration tests are added to cover multi-table validation scenarios.

  ---

  Issue 3: Update Documentation for v0.4.2 Changes

  Title: docs: Update usage.md and examples for v0.4.2 CLI changes

  Description:

  This issue covers updating all user-facing documentation to reflect the significant CLI changes and new features introduced in the v0.4.2 release. Clear
  documentation is critical for user adoption.

  Depends on: Completion of Issues #1 and #2.

  Acceptance Criteria:
   - [ ] All examples in docs/usage.md are rewritten to use the new --conn and --table argument format.
   - [ ] The new multi-table JSON format for the schema command is clearly documented with an example.
   - [ ] Any quick-start or usage examples in README.md are updated.
   - [ ] Files in the examples/ directory are reviewed and updated if necessary.
   - [ ] The output of vlite check --help and vlite schema --help is verified to be accurate and clear.
