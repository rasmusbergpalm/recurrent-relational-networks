import os
import tarfile
import urllib.request


class bAbI:
    url = "http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz"

    def __init__(self, folder):
        fname = "tasks_1-20_v1-2.tar.gz"
        dest = "/tmp/" + fname
        self.folder = folder

        if not os.path.exists(dest):
            print("Downloading data...")
            request = urllib.request.Request(self.url, None, {'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(request)
            data = response.read()
            with open(dest, 'wb') as f:
                f.write(data)

        with tarfile.open(dest) as tar:
            tar.extractall(path='/tmp')

        def load_tasks(set):
            def sanitize(line: str):
                return line.strip().lower().replace(".", "").replace("?", "")

            def parse(lines):
                lastid = -1
                facts = []
                questions = []
                for line in lines:
                    id, rest = line.split(' ', 1)
                    intid = int(id)
                    if intid < lastid:
                        facts = []
                    lastid = intid
                    if '\t' not in rest:
                        facts.append(rest)
                    else:
                        q, a, sf = rest.split('\t')
                        questions.append({
                            'q': q.strip(),
                            'a': a,
                            'facts': [f for f in facts]
                        })
                return questions

            tasks = []
            for i in range(1, 21):
                fname = '/tmp/tasks_1-20_v1-2/%s/qa%d_%s.txt' % (self.folder, i, set)

                with open(fname, 'r') as f:
                    lines = f.readlines()
                lines = [sanitize(line) for line in lines]
                tasks.append(parse(lines))

            return tasks

        self.train = load_tasks('train')
        self.valid = load_tasks('valid')
        self.test = load_tasks('test')
