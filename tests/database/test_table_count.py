def test_final_base_table_count_at_most_15(final_connection):
    with final_connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT count(*) FROM information_schema.tables
            WHERE table_type='BASE TABLE'
              AND table_schema IN ('system','catalog','ml','recommendation')
            """
        )
        assert cursor.fetchone()[0] == 13

        cursor.execute(
            """
            SELECT count(*) FROM pg_trigger t
            JOIN pg_class c ON c.oid=t.tgrelid
            JOIN pg_namespace n ON n.oid=c.relnamespace
            WHERE NOT t.tgisinternal
              AND n.nspname IN ('system','catalog','ml','recommendation')
            """
        )
        assert cursor.fetchone()[0] <= 2

        cursor.execute(
            """
            SELECT count(*) FROM pg_index i
            JOIN pg_class c ON c.oid=i.indrelid
            JOIN pg_namespace n ON n.oid=c.relnamespace
            WHERE NOT i.indisprimary
              AND n.nspname IN ('system','catalog','ml','recommendation')
            """
        )
        assert cursor.fetchone()[0] <= 20
