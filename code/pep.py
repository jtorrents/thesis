# -*- coding: utf-8 -*-
"""Code for handling object representation of a PEP.

   This belongs to pep0 from the peps official repository
   at https://hg.python.org/peps/
"""
import re
import textwrap
import unicodedata
from datetime import datetime

from email.parser import HeaderParser

from data import Author

class PEPError(Exception):

    def __init__(self, error, pep_file, pep_number=None):
        super(PEPError, self).__init__(error)
        self.filename = pep_file
        self.number = pep_number

    def __str__(self):
        error_msg = super(PEPError, self).__str__()
        if self.number is not None:
            return "PEP %d: %r" % (self.number, error_msg)
        else:
            return "(%s): %r" % (self.filename, error_msg)


class PEPParseError(PEPError):

    pass


class Author_orig(object):

    """Represent PEP authors.

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
        self.first_last = name.strip()
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
        return self.first_last.encode('UTF8', 'ignore')

class PEP(object):

    """Representation of PEPs.

    Attributes:

        + number : int
            PEP number.

        + title : str
            PEP title.

        + type_ : str
            The type of PEP.  Can only be one of the values from
            PEP.type_values.

        + status : str
            The PEP's status.  Value must be found in PEP.status_values.

        + authors : Sequence(Author)
            A list of the authors.
    """

    # The various RFC 822 headers that are supported.
    # The second item in the nested tuples represents if the header is
    # required or not.
    headers = (('PEP', True), ('Title', True), ('Version', True),
               ('Last-Modified', True), ('Author', True),
               ('BDFL-Delegate', False),
               ('Discussions-To', False), ('Status', True), ('Type', True),
               ('Content-Type', False), ('Requires', False),
               ('Created', True), ('Python-Version', False),
               ('Post-History', False), ('Replaces', False),
               ('Superseded-By', False), ('Resolution', False),
               )
    # Valid values for the Type header.
    type_values = (u"Standards Track", u"Informational", u"Process")
    # Valid values for the Status header.
    # Active PEPs can only be for Informational or Process PEPs.
    status_values = (u"Accepted", u"Rejected", u"Withdrawn", u"Deferred",
                     u"Final", u"Active", u"Draft", u"Superseded")

    def __init__(self, pep_file):
        """Init object from an open PEP file object."""
        # Parse the headers.
        self.filename = pep_file
        pep_parser = HeaderParser()
        metadata = pep_parser.parse(pep_file)
        header_order = iter(self.headers)
        try:
            for header_name in metadata.keys():
                current_header, required = header_order.next()
                while header_name != current_header and not required:
                    current_header, required = header_order.next()
                if header_name != current_header:
                    raise PEPError("did not deal with "
                                   "%r before having to handle %r" %
                                   (header_name, current_header),
                                   pep_file.name)
        except StopIteration:
            raise PEPError("headers missing or out of order",
                                pep_file.name)
        required = False
        try:
            while not required:
                current_header, required = header_order.next()
            else:
                raise PEPError("PEP is missing its %r" % (current_header,),
                               pep_file.name)
        except StopIteration:
            pass
        # 'PEP'.
        try:
            self.number = int(metadata['PEP'])
        except ValueError:
            raise PEPParseError("PEP number isn't an integer", pep_file.name)
        # 'Title'.
        self.title = metadata['Title']
        # 'Type'.
        type_ = metadata['Type']
        if type_ not in self.type_values:
            raise PEPError('%r is not a valid Type value' % (type_,),
                           pep_file.name, self.number)
        self.type_ = type_
        # 'Status'.
        status = metadata['Status']
        if status not in self.status_values:
            if status == "April Fool!":
                # See PEP 401 :)
                status = "Rejected"
            else:
                raise PEPError("%r is not a valid Status value" %
                               (status,), pep_file.name, self.number)
        # Special case for Active PEPs.
        if (status == u"Active" and
                self.type_ not in ("Process", "Informational")):
            raise PEPError("Only Process and Informational PEPs may "
                           "have an Active status", pep_file.name,
                           self.number)
        self.status = status
        # 'Author'.
        authors_and_emails = self._parse_author(metadata['Author'])
        if len(authors_and_emails) < 1:
            raise PEPError("no authors found", pep_file.name,
                           self.number)
        self.authors = map(Author, authors_and_emails)
        # 'BDFL-Delegate'
        self.delegates = []
        if 'BDFL-Delegate' in metadata:
            delegates_and_emails = self._parse_author(metadata['BDFL-Delegate'])
            self.delegates = map(Author, delegates_and_emails)
        # 'Created' for creation date of the PEP
        self.created = _parse_date(metadata['Created'])

    def _parse_author(self, data):
        """Return a list of author names and emails."""
        # XXX Consider using email.utils.parseaddr (doesn't work with names
        # lacking an email address.
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
        return author_list

    @property
    def type_abbr(self):
        """Return the how the type is to be represented in the index."""
        return self.type_[0].upper()

    @property
    def status_abbr(self):
        """Return how the status should be represented in the index."""
        if self.status in ('Draft', 'Active'):
            return u' '
        else:
            return self.status[0].upper()

    @property
    def author_abbr(self):
        """Return the author list as a comma-separated with only last names."""
        return u', '.join(x.nick for x in self.authors)

    @property
    def title_abbr(self):
        """Shorten the title to be no longer than the max title length."""
        if len(self.title) <= title_length:
            return self.title
        wrapped_title = textwrap.wrap(self.title, title_length - 4)
        return wrapped_title[0] + u' ...'

    def __unicode__(self):
        """Return the line entry for the PEP."""
        pep_info = {'type': self.type_abbr, 'number': str(self.number),
                'title': self.title_abbr, 'status': self.status_abbr,
                'authors': self.author_abbr}
        return column_format % pep_info


def _parse_date(date_str):
    if not date_str:
        return None
    if all(p.isdigit() for p in date_str.split('-')):
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            try:
                date = datetime.strptime(date_str, '%d-%m-%Y')
            except ValueError:
                date = None
    else:
        try:
            date = datetime.strptime(date_str, '%d-%b-%Y')
        except ValueError:
            try:
                date = datetime.strptime(date_str, '%d-%B-%Y')
            except ValueError:
                try:
                    date = datetime.strptime(date_str, '%d %B %Y')
                except ValueError:
                    try:
                        date = datetime.strptime(date_str, '%d %b %Y')
                    except ValueError:
                        date = None
    return date

title_length = 55
column_format = (u' %(type)1s%(status)1s %(number)4s  %(title)-' +
                    unicode(title_length) + u's %(authors)-s')

header = u"""PEP: 0
Title: Index of Python Enhancement Proposals (PEPs)
Version: N/A
Last-Modified: %s
Author: David Goodger <goodger@python.org>,
        Barry Warsaw <barry@python.org>
Status: Active
Type: Informational
Created: 13-Jul-2000
"""

intro = u"""
    This PEP contains the index of all Python Enhancement Proposals,
    known as PEPs.  PEP numbers are assigned by the PEP editors, and
    once assigned are never changed[1].  The Mercurial history[2] of
    the PEP texts represent their historical record.
"""

references = u"""
    [1] PEP 1: PEP Purpose and Guidelines
    [2] View PEP history online
        https://hg.python.org/peps/
"""

footer = u"""^L
Local Variables:
mode: indented-text
indent-tabs-mode: nil
sentence-end-double-space: t
fill-column: 70
coding: utf-8
End:"""

