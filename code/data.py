# -*- coding: utf-8 -*-
# This script is Python 2.x only as mercurial does not currently support
# python 3.x
import os
import re
import time
from optparse import OptionParser

from mercurial import ui, hg
import networkx as nx

from sna import utils

from project import multigraph_file, default_years, default_branches

repo_path = '/home/jtorrents/projects/cpython'
peps_path = '/home/jtorrents/projects/peps'

mail_re = re.compile("\<(?P<mail>.*)\>")

branches = ['2.0', '2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '2.7', 
            '3.0', '3.1', '3.2', '3.3', '3.4', 'default', 'legacy-trunk']

invalid_names = {'unknown', 'cvs2svn'}

names_equivalences = {
    u'Charles-Francois Natali': u'Charles-François Natali',
    u'Eric V. Smith': u'Eric Smith',
    u'Gerhard Haering': u'Gerhard Häring',
    u"Giampaolo Rodola'": u'Giampaolo Rodolà',
    u'guido@google.com': u'Guido van Rossum',
    u'Kristjan Valur Jonsson': u'Kristján Valur Jónsson',
    u'Marc-Andre Lemburg': u'Marc-André Lemburg',
    u'Martin v. Loewis': u'Martin v. Löwis',
    u'R David Murray': u'R. David Murray',
    u'Sean Reifschneider': u'Sean Reifscheider',
    u'Tarek Ziade': u'Tarek Ziadé',
    u'Terry Reedy': u'Terry Jan Reedy',
    u'Walter Doerwald': u'Walter Dörwald',
    u'Zbigniew JÄ™drzejewski-Szmek': u'Zbigniew Jędrzejewski-Szmek',
    u'brian.curtin': u'Brian Curtin',
    u'briancurtin': u'Brian Curtin',
    u'doko@python.org': u'Matthias Klose',
    u'doko@ubuntu.com': u'Matthias Klose',
    u'orsenthil@gmail.com': u'Senthil Kumaran',
}

##
## Author object
##
class Author(object):

    """Represent authors. This is based on pep0 code at pep.py.

    Attributes:

        + first_last : str
            The author's full name.

        + last_first : str
            Output the author's name in Last, First, Suffix order.

        + first : str
            The author's first name.  A middle initial may be included.

        + last : str
            The author's last name.

        + suffix : str
            A person's suffix (can be the empty string).

        + sort_by : str
            Modification of the author's last name that should be used for
            sorting.

        + email : str
            The author's email address.
    """

    def __init__(self, author_and_email_tuple):
        """Parse the name and email address of an author."""
        name, email = author_and_email_tuple
        first_last = name.strip()
        if first_last in names_equivalences:
            first_last = names_equivalences[first_last]
        self.first_last = first_last
        self.email = email.lower()
        last_name_fragment, suffix = self._last_name(name)
        name_sep = name.index(last_name_fragment)
        self.first = name[:name_sep].rstrip()
        self.last = last_name_fragment
        self.suffix = suffix
        if not self.first:
            self.last_first = self.last
        else:
            self.last_first = u', '.join([self.last, self.first])
            if self.suffix:
                self.last_first += u', ' + self.suffix
        if self.last == "van Rossum":
            # Special case for our beloved BDFL. :)
            if self.first == "Guido":
                self.nick = "GvR"
            elif self.first == "Just":
                self.nick = "JvR"
            else:
                raise ValueError("unkown van Rossum %r!" % self)
            self.last_first += " (%s)" % (self.nick,)
        else:
            self.nick = self.last

    def __hash__(self):
        return hash(self.first_last)

    def __eq__(self, other):
        return self.first_last == other.first_last

    @property
    def sort_by(self):
        name_parts = self.last.split()
        for index, part in enumerate(name_parts):
            if part[0].isupper():
                break
        else:
            raise ValueError("last name missing a capital letter: %r"
                                                           % name_parts)
        base = u' '.join(name_parts[index:]).lower()
        return unicodedata.normalize('NFKD', base).encode('ASCII', 'ignore')

    def _last_name(self, full_name):
        """Find the last name (or nickname) of a full name.

        If no last name (e.g, 'Aahz') then return the full name.  If there is
        a leading, lowercase portion to the last name (e.g., 'van' or 'von')
        then include it.  If there is a suffix (e.g., 'Jr.') that is appended
        through a comma, then drop the suffix.

        """
        name_partition = full_name.partition(u',')
        no_suffix = name_partition[0].strip()
        suffix = name_partition[2].strip()
        name_parts = no_suffix.split()
        part_count = len(name_parts)
        if part_count == 1 or part_count == 2:
            return name_parts[-1], suffix
        else:
            assert part_count > 2
            if name_parts[-2].islower():
                return u' '.join(name_parts[-2:]), suffix
            else:
                return name_parts[-1], suffix

    def __repr__(self):
        return self.first_last.encode('UTF8', errors='ignore')
        #return unicode(self.first_last, errors='ignore')


def _parse_mercurial_author(data, id_gen):
    """Also adapted from pep0 code to use with mercurial records"""
    angled = ur'(?P<author>.+?) <(?P<email>.+?)>'
    paren = ur'(?P<email>.+?) \((?P<author>.+?)\)'
    simple = ur'(?P<author>[^,]+)'
    author_list = []
    for regex in (angled, paren, simple):
        # Watch out for commas separating multiple names.
        regex += u'(,\s*)?'
        for match in re.finditer(regex, data):
            # Watch out for suffixes like 'Jr.' when they are comma-separated
            # from the name and thus cause issues when *all* names are only
            # separated by commas.
            match_dict = match.groupdict()
            author = match_dict['author']
            if not author.partition(' ')[1] and author.endswith('.'):
                prev_author = author_list.pop()
                author = ', '.join([prev_author, author])
            if u'email' not in match_dict:
                email = ''
            else:
                email = match_dict['email']
            author_list.append((author, email))
        else:
            # If authors were found then stop searching as only expect one
            # style of author citation.
            if author_list:
                break
    author = Author(author_list[0])
    user = author.first_last
    email = author.email
    uid = id_gen[user]
    return (uid, user, email)

##
## Build a bipartite multigraph from CPython's mercurial repository
##
def get_repo(path=repo_path):
    return hg.repository(ui.ui(), path)

def build_multigraph(repo=None, with_descriptions=False):
    if repo is None:
        repo = get_repo()
    M = nx.MultiGraph()
    M.graph['name'] = "CPython bipartite multigraph"
    id_gen = utils.UniqueIdGenerator()
    tip = repo.changelog.tip()
    for cid in xrange(repo.changelog.index[tip] + 1):
        if cid % 1000 == 0:
            print("changeset number {}".format(cid))
        cset = repo.changectx(cid)
        branch = cset.branch().decode('utf8')
        if with_descriptions:
            description = cset.description().decode('utf8')
        t = time.gmtime(cset.date()[0])
        year = t.tm_year
        day = time.strftime('%Y-%m-%d', t)
        hour = time.strftime('%H:%M:%S', t)
        #uid, user, email = _parse_user(cset.user(), id_gen)
        user_unicode = cset.user().decode('utf8', errors='ignore')
        uid, user, email = _parse_mercurial_author(user_unicode, id_gen)
        if user in invalid_names:
            continue
        if user not in M:
            M.add_node(user, bipartite=1, name=user, email=email, uid=uid)
        for fpath, diff in _iter_files_and_diffs(cset):
            fname = os.path.basename(fpath)
            fid = id_gen[fname]
            if fpath not in M:
                M.add_node(fpath, bipartite=0, name=fname, fullpath=fpath, fid=fid)
            added, deleted, weight = _parse_diff(diff)
            edge_attrs = {
                'branch': branch,
                'cid': cid,
                'hex': cset.hex(),
                'time': cset.date()[0],
                'year': year,
                'day': day,
                'hour': hour,
                'added': added,
                'deleted': deleted,
                'weight': weight,
            }
            if with_descriptions:
                edge_attrs['description'] = description
            M.add_edge(user, fpath, **edge_attrs)
    return M


def _iter_files_and_diffs(cset):
    # each file has two lines in the diff
    # header: diff -r f33ed9e2a607 -r 27cc7a314502 path/to/file.py\n
    # diff: the actual diff
    diff = cset.diff()
    for fpath in cset.files():
        fpath = fpath.decode('utf8')
        header = next(diff)
        actual_diff = next(diff)
        yield fpath, actual_diff


start_minus = re.compile("^-")
start_plus = re.compile("^\+")
def _parse_diff(diff):
    added = 0
    deleted = 0
    for line in diff.split('\n'):
        if start_plus.search(line):
            added += 1
        if start_minus.search(line):
            deleted += 1
    #return dict(added=added, deleted=deleted, weight=added+deleted)
    return (added, deleted, added+deleted)


def _parse_user(user_str, id_gen):
    user = user_str.split(' <')[0].strip()
    user = user.decode('utf8')
    uid = id_gen[user]
    m = mail_re.search(user_str)
    if m:
        return (uid, user, m.group('mail'))
    else:
        return (uid, user, 'Null')


##
## Build networks by year or by branch
##
multigraph = None
def get_multigraph():
    global multigraph
    if multigraph is None:
        if os.path.exists(multigraph_file):
            multigraph = nx.read_graphml(multigraph_file, node_type=unicode)
        else:
            multigraph = save_multigraph()
    return multigraph


def save_multigraph(M=None, fname=multigraph_file):
    if M is None:
        M = build_multigraph()
    nx.write_graphml(M, fname)
    return M


def build_graph_by_year(M, year):
    G = nx.Graph()
    G.graph['name'] = "CPython bipartite graph {}".format(year)
    G.graph['year'] = year
    for u, v in set(M.edges()):
        data = M.get_edge_data(u, v).values()
        relevant_data = [d for d in data if d['year']==year]
        if not relevant_data:
            continue
        G.add_edge(u, v,
            edits = len(relevant_data),
            weight = sum([d['weight'] for d in relevant_data]),
            added = sum([d['added'] for d in relevant_data]),
            deleted = sum([d['deleted'] for d in relevant_data]),
        )
    for n in G:
        G.node[n] = M.node[n]
    return G


def networks_by_year(M=None, years=default_years):
    if M is None:
        M = get_multigraph()
    for year in years:
        yield build_graph_by_year(M, year)


def build_graph_by_branch(M, branch, branches=branches):
    if branch not in branches:
        raise ValueError('Not a valid branch % s' % branch)
    G = nx.Graph()
    G.graph['name'] = "CPython bipartite graph {}".format(branch)
    G.graph['branch'] = branch
    for u, v in set(M.edges()):
        data = M.get_edge_data(u, v).values()
        relevant_data = [d for d in data if d['branch']==branch]
        if not relevant_data:
            continue
        G.add_edge(u, v,
            edits=len(relevant_data),
            weight = sum([d['weight'] for d in relevant_data]),
            added = sum([d['added'] for d in relevant_data]),
            deleted = sum([d['deleted'] for d in relevant_data]),
        )
    for n in G:
        G.node[n] = M.node[n]
    return G


def networks_by_branches(M=None, branches=default_branches):
    if M is None:
        M = get_multigraph()
    for branch in branches:
        yield build_graph_by_branch(M, branch)


##
## Process peps
##
def get_peps(path=peps_path):
    import codecs
    from operator import attrgetter
    from pep import PEP, PEPError

    peps = []
    if os.path.isdir(path):
        for file_path in os.listdir(path):
            if file_path == 'pep-0000.txt':
                continue
            abs_file_path = os.path.join(path, file_path)
            if not os.path.isfile(abs_file_path):
                continue
            if file_path.startswith("pep-") and file_path.endswith(".txt"):
                with codecs.open(abs_file_path, 'r', encoding='UTF-8') as pep_file:
                    try:
                        pep = PEP(pep_file)
                        if pep.number != int(file_path[4:-4]):
                            raise PEPError('PEP number does not match file name',
                                           file_path, pep.number)
                        peps.append(pep)
                    except PEPError, e:
                        errmsg = "Error processing PEP %s (%s), excluding:" % \
                            (e.number, e.filename)
                        print >>sys.stderr, errmsg, e
                        sys.exit(1)
        peps.sort(key=attrgetter('number'))
    elif os.path.isfile(path):
        with open(path, 'r') as pep_file:
            peps.append(PEP(pep_file))
    else:
        raise ValueError("argument must be a directory or file path")

    return [pep for pep in peps if pep.created is not None]


def main():
    parser = OptionParser()
    parser.add_option('-b','--build_multigraph',
                        action='store_true',
                        dest='build_multigraph',
                        help='Build and save CPython bipartite multigraph in graphml format.',
                        default=False)

    options, args = parser.parse_args()

    if options.build_multigraph:
        M = build_multigraph()
        M = save_multigraph(M=M)


if __name__ == '__main__':
    main()

