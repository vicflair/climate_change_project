import re

def parse_dirt():
    dirt_file = '../knowledge_base/simPath.lsp'
    with open(dirt_file, 'r') as f:
        dirt = f.readlines()
    paraphrases = {}
    current_relation = None
    for x in dirt:
        if '(sims' in x:
            current_relation = x[1:-7]
            paraphrases[current_relation] = []
        elif '))\n' in x:
            pass
        else:
            data = re.findall('[^\t\n]+', x)
            paraphrase = data[0]
            score = float(data[1])
            paraphrases[current_relation].append((paraphrase, score))
    return dirt


class DepRel:

    def __init__(self):
        self.root = True
        self.head = ''
        self.modifiers = []
        self.pos = ''

    def declare(self):
        print 'Root: ', self.root
        print 'Head: ', self.head
        print 'Modifiers: \n\t', self.print_modifiers()

    def set_root(self, x):
        self.root = x

    def set_head(self, x):
        self.head = x

    def add_modifier(self, x):
        x.set_root(False)
        self.modifiers.append(x)

    def set_modifiers(self, x):
        x.set_root(False)
        self.modifiers = x

    def print_modifiers(self):
        if self.modifiers:
            text = [m.head for m in self.modifiers]
            return '\n\t'.join(text)
        else:
            return ''

# Set up
verb = DepRel()
verb.set_head('causes')
obj = DepRel()
obj.set_head('storms')
adj = DepRel()
adj.set_head('severe')
obj.add_modifier(adj)
subj = DepRel()
subj.set_head('climate change')
verb.add_modifier(subj)
verb.add_modifier(obj)

# Test
verb.declare()
for m in verb.modifiers:
    print m.root