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
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path="/tmp")

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
