Carefully analyze the web app implemented in /server and /web for playing the trick taking card game [Sheepshead](Sheepshead.md) in real time against human and trained AI agents. I would like to:

  1. Clean up all the code so it is more maintainable, straightforward, and secure to prepare for releasing it.
  2. Update all dependencies to their latest versions.
  3. Add stats and game history tracking via postgres database, with the schema already defined in server/database/sheepshead-ai-db-schema.sql.
  I would like to add and maintain this schema using the graphile migrate library: https://github.com/graphile/migrate
  4. As part of the above update to record games in postgres, I would like to automatically generate player UUIDs for any player who joins a
  table and store this ID in their browser's local storage. If (and only if) they change their name from one of the pre-populated defaults, we
  should also store their name value in the database and their local storage and use it for future games. If they do not have a manually-chosen name, it should keep using the existing random default (which changes on every reload). A future update may allow players to create accounts to sign in, but that is out of scope for now.


Please create a detailed project plan for this series of updates that can be competently followed by a junior engineer and save it to a markdown file. If you have any questions whatsoever about ambiguities in the database schema and how we should save any part of the data, please ask for clarification before making assumptions.
