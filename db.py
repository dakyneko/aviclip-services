import sqlite3

class SQLiteDB(object):

    def __init__(self, path, name, col_types):
        '''assumes first column in col_types is the unique id'''
        self.path = path
        self.name = name

        self.execute('pragma journal_mode = WAL')
        self.execute('pragma mmap_size = 30000000000')

        self._create_table(col_types)
        self.cols = list(zip(*col_types))[0]

        self.insert_query = 'INSERT OR IGNORE INTO %s VALUES (%s)' % (self.name, ', '.join(['?'] * len(self.cols)))
        self.update_query = 'UPDATE %s SET %s WHERE %s = ?' % (self.name, self.unique_col, ', '.join(['%s = ?' % col for col in self.cols[1:]]))

        self._create_index()

    def _create_index(self):
        self.execute('CREATE INDEX IF NOT EXISTS %s_index on %s (%s)' % (self.name, self.name, self.unique_col))

    def _create_table(self, col_types):
        self.execute('CREATE TABLE IF NOT EXISTS %s (%s)' % (self.name, ', '.join(['%s %s %s' % (c, t, 'UNIQUE' if i == 0 else '') for i, (c, t) in enumerate(col_types)])))

    @property
    def unique_col(self):
        return self.cols[0]

    def _execute(self, *args, chunk_size=1000):
        with sqlite3.connect(self.path, timeout=1000) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            if type(args[-1]) == types.GeneratorType:
                cur.executemany(*args)
            else:
                cur.execute(*args)
            con.commit()
            while True:
                results = cur.fetchmany(chunk_size)
                if len(results) == 0:
                    break
                for result in results:
                    yield result

    def execute(self, *args, generator=False, **kwargs):
        results = self._execute(*args, **kwargs)
        if not generator:
            results = list(results)
        return results

    def safely_execute(self, *args):
        try:
            self.execute(*args)
            return True
        except sqlite3.IntegrityError:
            return False

    def get(self, rowid):
        results = self.execute('SELECT * FROM %s WHERE rowid = ?' % self.name, (rowid,))
        return Bunch(results[0])

    def search(self, text, col):
        results = self.execute('SELECT * from %s WHERE %s MATCH ?' % (self.name, col), (text,))
        for result in results:
            yield(Bunch(result))

    def size(self):
        results = self.execute('SELECT COUNT(*) FROM %s' % self.name)
        return results[0][0]

    def iter_rows(self):
        rows = self.execute('SELECT * FROM %s' % self.name, generator=True)
        for row in rows:
            yield(Bunch(row))

    def contains(self, elt):
        results = self.execute('SELECT COUNT(*) FROM %s WHERE %s = ?' % (self.name, self.unique_col), (elt[self.unique_col],))
        count = results[0][0]
        return count == 1

    def _iter_insertable_elts(self, elts):
        for elt in elts:
            yield(tuple([elt.get(col, None) for col in self.cols]))

    def insert(self, elts):
        return self.safely_execute(self.insert_query, self._iter_insertable_elts(elts))

    def update(self, elt):
        return self.safely_execute(self.update_query, tuple([elt.get(col, None) for col in self.cols[1:]] + [elt[self.unique_col]]))
