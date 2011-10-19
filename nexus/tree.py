#!/usr/bin/env python
# This program is public domain

"""
Tree view for NeXus files.

The `nexus.tree` routines provide a natural interface to NeXus datasets.
Entries in a group are referenced much like fields in a class are
referenced in python.  Rather than following the directory model of
the `nexus.napi` interface, users are free to reference separate fields
in the dataset at the same time.  Large datasets are not read until
they are needed, and may be read or written one slab at a time.

There are a number of functions which operate on files::

    import nexus
    nxfile = nexus.load('file.nxs') # loads a structure from a file
    nexus.save('copy.nxs', tree)    # saves a structure to a file
    nexus.tree('copy.nxs')          # display the contents of a file

The tree returned from load() has an entry for each group, field and
attribute.  You can traverse the hierarchy using the names of the
groups.  For example, tree.entry.instrument.detector.distance is an
example of a field containing the distance to each pixel in the
detector. Entries can also be referenced by NXclass name, such as
tree.NXentry[0].instrument. Since there may be multiple entries of the
same NXclass, the NXclass attribute returns a possibly empty list.

Properties of the entry in the tree are referenced by nx attributes.
Depending on the node type, different nx attributes may be available.

Nodes (class NXnode) have attributes shared by both groups and fields::
    * nxname     node name
    * nxclass    node class for groups, 'SDS' for fields
    * nxgroup    group containing the entry, or None for the root
    * nxattrs    dictionary of NeXus attributes for the node

Groups (class NXgroup) have attributes for accessing children::
    * nxentries  dictionary of entries within the group
    * nxcomponent('nxclass')  return group entries of a particular class
    * nxdir()    print the list of entries in the group
    * nxtree()   print the list of entries and subentries in the group
    * nxplot()   plot signal and axes for the group, if available

Fields (class SDS) have attributes for accessing data:
    * nxdims     dimensions of data in the field
    * nxtype     data type
    * nxdata     data in the field
    * nxdata_as('units')      get data in particular units
    * nxdata_from(value,'units')  set data in particular units
    * nxslab     slab context for the field

Linked fields (class NXlink) have attributes for accessing the link::
    * nxlink     reference to the linked field

Unknown fields (class Unknown) are groups with a name that doesn't
start with 'NX'.  These groups are not loaded or saved.

NeXus attributes (class NXattr) have a type and a value only::
    * nxtype     attribute type
    * nxdata     attribute data

Data can be stored in the NeXus file in a variety of units, depending
on which facility is storing the file.  This makes life difficult
for reduction and analysis programs which must know the units they
are working with.  Our solution to this problem is to allow the reader
to retrieve data from the file in particular units.  For example, if
detector distance is stored in the file using millimeters you can
retrieve them in meters using::
    entry.instrument.detector.distance.nxdata_as('m')
See `nexus.unit` for more details on the unit formats supported.

The slab interface to field data works by opening the file handle
and keeping it open as long as the slab interface is needed.  This
is done in python 2.5 using the with statement.  Once the context
is entered, nxget() and nxput() methods on the node allow you to read
and write data a slab at a time.  For example::

    # Read a Ni x Nj x Nk array one vector at a time
    with root.NXentry[0].data.data as slab:
        Ni,Nj,Nk = slab.nxdims
        size = [1,1,Nk]
        for i in range(Ni):
            for j in range(Nj):
                value = slab.nxget([i,j,0],size)

The equivalent can be done in Python 2.4 and lower using the context
functions __enter__ and __exit__::

    slab = data.nxslab.__enter__()
    ... do the slab functions ...
    data.nxslab.__exit__()

You can traverse the tree by component class instead of component name.
Since there may be multiple components of the same class in one group
you will need to specify which one to use.  For example,
tree.NXentry[0].NXinstrument[0].NXdetector[0].distance references the
first detector of the first instrument of the first entry.  Unfortunately,
there is no guarantee regarding the order of the entries, and it may vary
from call to call, so the above is of limited utility.

The nxplot() method for groups uses matplotlib to plot the data.  You
can replace this with your own plotter by setting nexus.NXgroup._plotter
to your own plotter class.  The plotter class has one method::

    plot(signal, axes, entry, title)

where signal is the field containing the data, axes are the fields
listing the signal sample points, entry is file/path within the file
to the data group and title is the title of the NXentry, if available.

The load() and save() functions are implemented using the class
`nexus.tree.NeXusTree`, a subclass of `nexus.napi.NeXus` which allows
all the usual API functions.  You can subclass NeXusTree with your
own version that defines, e.g., a NXmonitor() method to return an
NXmonitor object when an NXmonitor class is read.  Your NXmonitor
class should probably be a subclass of NXgroup.
"""
__all__ = ['load', 'save', 'tree', 'NeXusTree']

from copy import copy, deepcopy
import re

import numpy
from . import napi, unit
from .napi import NeXusError


# Maximum memory in MB
NX_MAX_MEMORY = 500

class NeXusTree(napi.NeXus):
    """
    Structure-based interface to the NeXus file API.

    Usage::

      file = NeXusTree(filename, ['r','rw','w'])
        - open the NeXus file
      root = file.readfile()
        - read the structure of the NeXus file.  This returns a NeXus tree.
      file.writefile(root)
        - write a NeXus tree to the file.
      data = file.readpath(path)
        - read data from a particular path

    Example::

      nx = NeXusTree('REF_L_1346.nxs','r')
      tree = nx.readfile()
      for entry in tree.NXentry:
          process(entry)
      copy = NeXusTree('modified.nxs','w')
      copy.writefile(tree)

    Note that the large datasets are not loaded immediately.  Instead, the
    when the data set is requested, the file is reopened, the data read, and
    the file closed again.  open/close are available for when we want to
    read/write slabs without the overhead of moving the file cursor each time.
    The NXdata nodes in the returned tree hold the node values.

    Subclasses can provide methods for individual NeXus classes such
    as NXbeam or NXdata.  Brave users can also specialize NXgroup,
    NXattr, SDS and NXlink methods.
    """
    def readfile(self):
        """
        Read the nexus file structure from the file.  Large datasets
        are not read until they are needed.

        Returns a tree of NXgroup, SDS and NXlink nodes.
        """
        self.open()
        self.openpath("/")
        root = self._readgroup()
        self.close()
        root.nxgroup = None

        # Resolve links
        self._readlinks(root)
        return root

    def writefile(self, tree):
        """
        Write the nexus file structure to the file.  The file is assumed to
        start empty.

        Updating individual nodes can be done using the napi interface, with
        nx.handle as the nexus file handle.
        """
        self.open()
        links = []
        # Root node is special --- only write its children.
        # TODO: maybe want to write root node attributes?
        for entry in tree.nxentries.values():
            links += self._writegroup(entry, path="")
        self._writelinks(links)
        self.close()

    def writenode(self, node, path=""):
        """
        Write a node to a nexus file.
        """
        self.open()
        self._writelinks(self._writegroup(entry, path=path))
        self.close()

    def readpath(self, path):
        """
        Read the data on a particular file path.

        Returns a numpy array containing the data, a python scalar, or a
        string depending on the shape and storage class.
        """
        self.open()
        self.openpath(path)
        return self.getdata()

    def _readattrs(self):
        """
        Return the attributes for the currently open group/data or for
        the file if no group or data object is open.
        """
        attrs = dict()
        for i in range(self.getattrinfo()):
            name,length,nxtype = self.getnextattr()
            value = self.getattr(name, length, nxtype)
            pair = self.NXattr(value,nxtype)
            attrs[name] = pair
            #print "read attr",name,pair.nxdata, pair.nxtype
        return attrs

    def _readdata(self, name):
        """
        Read a data node, returning SDS or NXlink depending on the
        nature of the node.
        """
        # Finally some data, but don't read it if it is big
        # Instead record the location, type and size
        self.opendata(name)
        attrs = self._readattrs()
        if 'target' in attrs and attrs['target'].nxdata != self.path:
            # This is a linked dataset; don't try to load it.
            #print "read link %s->%s"%(path,attrs['target'].nxdata)
            data = NXlink(name,nxclass='SDS',attrs=attrs)
        else:
            dims,type = self.getinfo()
            if numpy.prod(dims) < 1000: # Fewer than 1000 elements
                value = self.getdata()
            else:
                value = None
            data = self.SDS(value=value,name=name,dtype=type,shape=dims,attrs=attrs,
                            file=self,path=self.path)
        self.closedata()
        return data

    # These are groups that HDFView explicitly skips
    _skipgroups = ['CDF0.0','_HDF_CHK_TBL_','Attr0.0','RIG0.0','RI0.0',
                   'RIATTR0.0N','RIATTR0.0C']

    def _readchildren(self,n):
        children = {}
        for _ in range(n):
            name,nxclass = self.getnextentry()
            #print "name,class,path",name,nxclass,self.path
            if nxclass in self._skipgroups:
                pass # Skip known bogus classes
            elif nxclass == 'SDS':
                children[name] = self._readdata(name)
            elif nxclass.startswith('NX'):
                self.opengroup(name,nxclass)
                children[name] = self._readgroup()
                self.closegroup()
            else: # Bad entry
                ## flag it but don't do anything
                #children[name] = self.Unknown(name=name,nxclass=nxclass)
                # Read it anyway
                self.opengroup(name,nxclass)
                children[name] = self._readgroup()
                self.closegroup()
        return children

    def _readgroup(self):
        """
        Read the currently open group and all subgroups.
        """
        # TODO: does it make sense to read without recursing?
        # TODO: can we specify which NXclasses we are interested
        # in and skip those of different classes?
        n,name,nxclass = self.getgroupinfo()
        attrs = self._readattrs()
        if 'target' in attrs and attrs['target'].nxdata != self.path:
            # This is a linked group; don't try to load it.
            #print "read group link %s->%s"%(attrs['target'].nxdata,self.path)
            group = self.NXlink(name,nxclass=nxclass,attrs=attrs)
        else:
            #print "read group",nxclass,"from",self.path
            children = self._readchildren(n)
            # If we are subclassed with a handler for the particular
            # NXentry class name use that constructor for the group
            # rather than the generic NXgroup class.
            if hasattr(self,nxclass):
                factory = getattr(self,nxclass)
            else:
                factory = self.NXgroup
            group = factory(nxclass=nxclass,name=name,attrs=attrs,entries=children)
            # Build chain back structure
            for node in children.values():
                node.nxgroup = group
        return group

    def _readlinks(self, root):
        """
        Convert linked nodes into direct references.
        """
        for entry in root.nxentries.values():
            if isinstance(entry, NXlink):
                link = root
                for level in self._link_target.split('.'):
                    link = getattr(link,level)
                entry.nxlink = link

    # Allow subclasses to override
    def NXattr(self, *args, **kw): return NXattr(*args, **kw)
    def SDS(self, *args, **kw): return SDS(*args, **kw)
    def NXgroup(self,*args,**kw): return NXgroup(*args, **kw)
    def NXlink(self, *args, **kw): return NXlink(*args, **kw)
    def Unknown(self, *args, **kw): return Unknown(*args, **kw)

    def _writeattrs(self, attrs):
        """
        Return the attributes for the currently open group/data or for
        the file if no group or data object is open.
        """
        for name,pair in attrs.iteritems():
            #print "write attrs",name,pair.nxtype,pair.nxdata
            self.putattr(name,pair.nxdata,pair.nxtype)

    def _writedata(self, data, path):
        """
        Write the given data node.

        NXlinks cannot be written until the linked group is created, so
        this routine returns the set of links that need to be written.
        Call writelinks on the list.
        """

        path = path + "/" + data.nxname
        #print 'write data',path

        # If the data is linked then
        if hasattr(data,'_link_target'):
            return [(path, data._link_target)]

        # Finally some data.  Compress it if it is large.
        #print "creating data",data.nxname,data.nxdims,data.nxtype
        if numpy.prod(data.nxdims) > 10000:
            # Compress the fastest moving dimension of large datasets
            slab_dims = numpy.ones(len(data.nxdims),'i')
            slab_dims[-1] = data.nxdims[-1]
            self.compmakedata(data.nxname, data.nxtype, data.nxdims,
                              'lzw', slab_dims)
        else:
            # Don't use compression for small datasets
            self.makedata(data.nxname, data.nxtype, data.nxdims)
        self.opendata(data.nxname)
        self._writeattrs(data.nxattrs)
        value = data.nxdata
        if value is not None:
            self.putdata(data.nxdata)
        self.closedata()
        return []

    def _writegroup(self, group, path):
        """
        Write the given group structure, including the data.

        NXlinks cannot be written until the linked group is created, so
        this routine returns the set of links that need to be written.
        Call writelinks on the list.
        """
        path = path + "/" + group.nxname
        #print 'write group',path

        links = []
        self.makegroup(group.nxname, group.nxclass)
        self.opengroup(group.nxname, group.nxclass)
        self._writeattrs(group.nxattrs)
        if hasattr(group, '_link_target'):
            links += [(path, group._link_target)]
        for child in group.nxentries.values():
            if child.nxclass == 'SDS':
                links += self._writedata(child,path)
            elif hasattr(child,'_link_target'):
                links += [(path+"/"+child.nxname,child._link_target)]
            else:
                links += self._writegroup(child,path)
        self.closegroup()
        return links

    def _writelinks(self, links):
        """
        Create links within the NeXus file as indicated by the set of pairs
        returned by writegroup.
        """
        gid = {}

        # identify targets
        for path,target in links:
            gid[target] = None

        # find gids for targets
        for target in gid.iterkeys():
            #sprint "target",target
            self.openpath(target)
            # Can't tell from the name if we are linking to a group or
            # to a dataset, so cheat and rely on getdataID to signal
            # an error if we are not within a group.
            try:
                gid[target] = self.getdataID()
            except NeXusError:
                gid[target] = self.getgroupID()

        # link sources to targets
        for path,target in links:
            if path != target:
                # ignore self-links
                parent = "/".join(path.split("/")[:-1])
                #print "link %s -> %s"%(parent,target)
                self.openpath(parent)
                self.makelink(gid[target])

_pythontype={int:'int32', long:'int64', float:'float64'}
_nxtype_size={'char':1, 'float32':4, 'float64':8,
              'int8':1, 'uint8':1, 'int16':2, 'uint16':2,
              'int32':4, 'uint32':4, 'int64':8, 'uint64':8}

def _datasize(nxtype,dims):
    return _nxtype_size[nxtype]*numpy.prod(dims)/(1024**2)

def _convert_type(value, nxtype=None):
    # Convert string-like inputs to strings

    try:
        if value != '': value[0]+''
        if nxtype in (None, '', 'char'):
            value = str(value)
            return 'char',[len(value)],value
    except:
        if nxtype == 'char':
            raise TypeError("Expected a string")

    # Convert numeric inputs to numpy arrays
    if nxtype == '':
        value = numpy.asarray(value)
        if str(value.dtype) not in _nxtype_size:
            raise TypeError("Expected a string or array")
    else:
        #print "converting",getattr(value,'dtype',type(value)),"to",nxtype
        value = numpy.asarray(value, nxtype)
    if numpy.isscalar(value):
        value = value.reshape((1,))
    return str(value.dtype),value.shape,value

_AXES_SEP=re.compile('[\[]*(\s*,*:*)+[\]]*')
def _parse_axes(axes):
    """
    Parse the 'axes' attribute for the axis names.

    The delimiter separating each axis can be white space, a comma, or a colon.
    """
    return [x for x in _AXES_SEP.split(axes) if len(x)>0]


class NXattr(object):
    """
    Attributes need to keep track of nxtype as well as attribute value.
    """
    def __init__(self,value=None,dtype=None):
        if isinstance(value, NXattr):
            self.nxdata,self.nxtype = value.nxdata,value.nxtype
        else:
            self.nxtype,_,self.nxdata = _convert_type(value,dtype)

    def __str__(self):
        return str(self.nxdata)

    def __repr__(self):
        if self.nxtype == 'char':
            return "NXattr('%s','%s')"%(self.nxdata,self.nxtype)
        else:
            return "NXattr(%s,'%s')"%(self.nxdata,self.nxtype)

    def __eq__(self, other):
        """
        Return true if the value of the attribute is the same as the other.
        """
        if isinstance(other, NXattr):
            return self.nxdata == other.nxdata
        else:
            return self.nxdata == other


class NXnode(object):
    """
    Abstract base class for elements in NeXus files.

    The node has a subclass of SDS (Scientific Data Set), NXgroup, or one
    of the NXgroup subclasses. Child nodes should be accessible directly as
    object attributes. Constructors for NXnode objects are defined by either
    the SDS or NXgroup classes.

    Attributes
    ----------
    nxclass : string
        The class of the NXnode. NXnodes can have class SDS, NXgroup,
        or one of the NXgroup subclasses.
    nxname : string
        The name of the NXnode.
    nxgroup : NXnode
        The parent NXgroup, if the node is defined as the attribute of parent
        node.
    nxentries : list
        A list of all the NeXus objects contained within an NXgroup. This list
        excludes all node attributes whose names begin with 'nx' or '_'.
    nxattrs : list
        A list of all the NeXus attributes, i.e., attribute with class NXattr.
    nxpath : string
        The path to this node with respect to the root of the NeXus tree. For
        NeXus data read from a file, this will be a group of class NXroot, but
        if the NeXus tree was defined interactively, it can be any valid
        NXgroup.
    """
    nxclass = "unknown"
    nxname = "unknown"
    nxgroup = None

    def __str__(self):
        return "%s:%s"%(self.nxclass,self.nxname)

    def __repr__(self):
        return "NXnode('%s','%s')"%(self.nxclass,self.nxname)

    def _setattrs(self, attrs):
        for k,v in attrs.items():
            setattr(self, k, v)

    def _attrs(self):
        return dict([(k,v)
                     for k,v in self.__dict__.items()
                     if isinstance(v,NXattr)])

    def _entries(self):
        return dict([(k,v)
                     for k,v in self.__dict__.items()
                     if isinstance(v,NXnode) and not k.startswith('nx')
                        and not k.startswith('_')])

    nxattrs = property(_attrs,doc="NeXus attributes for node")
    nxentries = property(_entries,doc="NeXus nodes within group")

    def _str_name(self,indent=0):
        if self.nxclass == 'SDS':
            return " "*indent+self.nxname
        else:
            return " "*indent+self.nxname+':'+self.nxclass

    def _str_value(self,indent=0):
        return ""

    def _str_attrs(self,indent=0):
        attrs = self.nxattrs
        names = attrs.keys()
        names.sort()
        return "\n".join(" "*indent+"@%s = %s"%(k,attrs[k].nxdata) for k in names)

    def _str_tree(self,indent=0,attrs=False,recursive=False):
        """
        Print current node and possibly subnodes.
        """
        # Print node
        result = [self._str_name(indent=indent)]
        if attrs:
            result.append(self._str_attrs(indent=indent+2))
        result.append(self._str_value(indent=indent+2))
        # Print children
        entries = self.nxentries
        names = entries.keys()
        names.sort()
        if recursive:
            result.extend(entries[k]._str_tree(indent=indent+2,
                                               attrs=attrs, recursive=True)
                          for k in names)
        else:
            result.extend(entries[k]._str_name(indent=indent+2)
                          for k in names)
        return "\n".join(l for l in result if l)

    def nxdir(self,attrs=False,recursive=False):
        """
        Print the node directory.

        The directory is a list of NeXus objects within this node, either
        NeXus groups or SDS data. If 'attrs' is True, SDS attributes are
        displayed. If 'recursive' is True, the contents of child groups are
        also displayed.
        """
        print self._str_tree(attrs=attrs,recursive=recursive)

    def nxtree(self,attrs=True):
        """
        Print the directory tree.

        The tree contains all child objects of this node and their children.
        It invokes the 'nxdir' method with both 'attrs' and 'recursive' set
        to True.
        """
        print self._str_tree(attrs=attrs,recursive=True)

class SDS(NXnode):
    """
    A NeXus data node (Scientific Data Set).

    This is a subclass of NXnode to contain scalar, array, or string data
    and associated NeXus attributes.

    SDS(value=None, name='unknown', dtype='', shape=[], attrs={}, file=None,
        path=None, group=None, **attr)

    Parameters
    ----------
    value : scalar value, Numpy array, or string
        The numerical or string value of the SDS, which is directly accessible
        as the SDS attribute 'nxdata'.
    name : string
        The name of the SDS, which is directly accessible as the SDS
        attribute 'nxname'. If the SDS is initialized as the attribute of
        a parent node, the name is automatically set to the name of this
        attribute.
    dtype : string
        The data type of the SDS value, which is directly accessible as the
        SDS attribute 'nxtype'. Valid types correspond to standard Numpy
        data types, using names defined by the NeXus API,
        i.e., 'float32' 'float64'
              'int8' 'int16' 'int32' 'int64'
              'uint8' 'uint16' 'uint32' 'uint64'
              'char'
        If the data type is not specified, then it is determined automatically
        by the data type of the 'value' parameter.
    shape : list of ints
        The dimensions of the SDS data, which is accessible as the SDS
        attribute 'nxdims'. This corresponds to the shape of a Numpy array,
        or the length of a string. The shape is [1] if the value is a scalar.
    attrs : dict
        A dictionary containing SDS attributes. The dictionary values should
        all have class NXattr.
    file : filename
        The file from which the SDS has been read.
    path : string
        The path to this node with respect to the root of the NeXus tree,
        using the convention for unix file paths.
    group : NXnode (SDS, NXgroup, or subclass of NXgroup)
        The parent NeXus node, which is accessible as the SDS attribute
        'nxgroup'. If the SDS is initialized as the attribute of
        a parent node, the group is set to the parent node.

    SDS attributes can also be set directly by keyword arguments, which are
    converted to objects of class NXattr, e.g.,

        temperature = SDS(40.0, units='K')

    Attributes
    ----------
    nxclass : 'SDS'
        The class of the NXnode.
    nxname : string
        The name of the SDS.
    nxtype : string
        The data type of the SDS value. Valid values are given above.
    nxdims : list of ints
        The dimensions of the SDS data. This is equivalent to the shape of a
        Numpy array.
    nxattrs : dict
        A dictionary of the SDS attributes, i.e., those with class NXattr
    nxdata : scalar, Numpy array or string
        The data value of the SDS. This is normally initialized using the
        'value' parameter (see above). If the NeXus data is contained
        in a file and the size of the SDS array is too large to be stored
        in memory, the value is not read in until this attribute is directly
        accessed. Even then, if there is insufficient memory, a value of None
        will be returned. In this case, the SDS array should be read as a
        series of smaller slabs using 'nxget'.
    nxdata_as('units') : scalar value or Numpy array
        If the SDS 'units' attribute has been set, the data values, stored
        in 'nxdata', are returned after conversion to the specified units.
    nxdata_from(value, 'units')
        Update the nxdata value with a new value in the given units
    nxpath : string
        The path to this node with respect to the root of the NeXus tree. For
        NeXus data read from a file, this will be a group of class NXroot, but
        if the NeXus tree was defined interactively, it can be any valid
        NXgroup. This is determined by recursively accessing the 'nxgroup'
        attributes of the parent nodes.

    Methods
    -------
    nxdir(self, attrs=False):
        Print the SDS specification.

        This outputs the name, dimensions and data type of the SDS.
        If 'attrs' is True, SDS attributes are displayed.

    nxtree(self, attrs=True):
        Print the SDS and its attributes.

        It invokes the 'nxdir' method with 'attrs' set to True.

    nxsave(filename, format='w5')
        Save the SDS into a file wrapped in a NXroot group and NXentry group
        with default names. This is equivalent to

        >>> NXroot(NXentry(SDS(...))).nxsave(filename)

    Examples
    --------
    >>> x = SDS(np.linspace(0,2*np.pi,101), units='degree')
    >>> phi = x.nxdata_as(units='radian')
    >>> y = SDS(np.sin(phi))

    # Read a Ni x Nj x Nk array one vector at a time
    >>> with root.NXentry[0].data.data as slab:
            Ni,Nj,Nk = slab.nxdims
            size = [1,1,Nk]
            for i in range(Ni):
                for j in range(Nj):
                    value = slab.nxget([i,j,0],size)

    """
    def __init__(self, name=None, dtype='', shape=(), attrs={},
                 file=None, path=None, nxgroup=None,
                 value=None, units=None):
        #print "creating data node for",path
        self._file = file
        self._path = path
        self._value = None
        self._dirty = False
        self.nxclass = "SDS" # Sciefntific Data Set
        self.nxname = name
        self.nxtype = str(dtype)
        self.nxdims = shape
        self.nxgroup = nxgroup
        # Convert NeXus attributes to python attributes
        self._setattrs(attrs)
        # If units appears as a keyword, it overrides any units given in attrs.
        if units is not None:
            self.nxunits = units
        self._converter = unit.Converter(self.nxunits)
        self._incontext = False
        if value is not None:
            self.nxdata = value

    def __enter__(self):
        """
        Open the datapath for reading slab by slab.

        Note: the results are undefined if you try accessing
        more than one slab at a time.  Don't nest your
        "with data" statements!
        """
        # TODO: provide a file lock to prevent movement of the
        # file cursor when in the slab context.
        # TODO: if HDF allows multiple cursors, extend napi to support them
        self._close_on_exit = not self._file.isopen
        self._file.open() # Force file open even if closed
        self._file.openpath(self._path)
        self._incontext = True
        return self

    def __exit__(self, *args):
        """
        Close the file associated with the data after reading.
        """
        self._incontext = False
        if self._close_on_exit:
            self._file.close()

    def nxget(self, offset, size, units=None):
        """
        Read a slab from the file.

        Offsets are 0-origin.  Shape can be inferred from the data.
        Offset and shape must each have one entry per dimension.

        If units are specified, convert the values to the given units
        before returning them.

        This operation should be performed in a "with group.data"
        context.

        Raises ValueError cannot convert units.

        Corresponds to NXgetslab(handle,data,offset,shape)
        """
        if self._file is None:
            raise IOError("Data is not attached to a file")
        if self._dirty or self._value:
            raise IOError("Cannot mix nxget and nxdata_as/nxdata_from")
        with self:
            value = self._file.getslab(offset,size)
        return self._converter(value,units)

    def nxput(self, data, offset, units=None):
        """
        Write a slab into the file.

        Offsets are 0-origin.  Shape can be inferred from the data.
        Offset and shape must each have one entry per dimension.

        This operation should be performed in a "with group.data"
        context.

        Raises ValueError if this fails.  No error is raised when
        writing to a file which is open read-only.

        Corresponds to NXputslab(handle,data,offset,shape)

        Note that this operation clears the nxdata value, forcing it to
        be reloaded the next time it is accessed.
        """
        if self._file is None:
            raise IOError("Data is not attached to a file")
        if self._dirty:
            raise IOError("Cannot mix nxput and nxdata_from")
        if isinstance(data, SDS):
            # When copying from one SDS to another, convert to the units
            # of the target SDS.
            data = data.nxdata_as(self.nxunits)
        elif units:
            # When storing a block of data, convert to the units of the SDS
            data = unit.Converter(units)(data, self.nxunits)
        with self:
            self._file.putslab(numpy.asarray(data,self.nxtype), offset, data.shape)
        self._value = None # Clear the cached value, if any; it is no longer valid

    def nxadd(self, data, offset, units=""):
        """
        Increment a slab in the data array.

        Calls nxget to read in existing data before adding the value
        and calling nxput. It assumes that the two sets of data have
        compatible data types.
        """
        if isinstance(data, SDS):
            # When adding one SDS to another, first convert to target SDS units
            data = data.nxdata_as(self.nxunits)
        elif units:
            # When adding to an SDS, first convert to the SDS units
            data = unit.Converter(units)(data, self.nxunits)

        current_value = self.nxget(offset, data.shape)
        self.nxput(current_value + data, offset)


    def __str__(self):
        """
        If value is loaded, return the value as a string.  If value is
        not loaded, return the empty string.  Only the first view values
        for large arrays will be printed.
        """
        if self._value is not None:
            return str(self._value)
        return ""

    def _str_value(self,indent=0):
        v = str(self)
        if '\n' in v:
            v = '\n'.join([(" "*indent)+s for s in v.split('\n')])
        return v

    def _str_tree(self,indent=0,attrs=False,recursive=False):
        dims = 'x'.join([str(n) for n in self.nxdims])
        #return "%s(%s)"%(self.nxtype, dims)
        s = str(self)
        if '\n' in s or s == "":
            s = "%s(%s)"%(self.nxtype, dims)
        v=[" "*indent + "%s = %s"%(self.nxname, s)]
        if attrs and self.nxattrs: v.append(self._str_attrs(indent=indent+2))
        return "\n".join(v)

    def nxaxes(self):
        """
        Return a list of SDSs containing axes.

        Only works if the SDS has the 'axes' attribute
        """
        return [getattr(self.nxgroup, name) for name in _parse_axes(self.axes.nxdata)]

    def nxdata_as(self, units=""):
        """
        Return the data in particular units.
        """
        if self._value is None:
            if not self._file: return None
            if _datasize(self.nxtype, self.nxdims) > NX_MAX_MEMORY:
                raise ValueError("Data larger than NX_MAX_MEMORY")
            self._value = self._file.readpath(self._path)

        #print "retrieving",self.nxname,"as",getattr(self._value,'dtype',type(self._value))
        if units:
            return self._converter(self._value,units)
        else:
            return self._value

    def nxdata_scale(self, units=""):
        """
        Return the scale required to convert from stored data to the
        particular units.
        """
        return self._converter(1.,units)

    def nxdata_from(self, value, units=""):
        """
        Replace the data with a new value.  The data is converted to the
        appropriate type and units.  Note that array dimensions are not
        checked.  Any consistency requirements, such as the time of flight
        matching the last dimension of the detector, must be checked by
        the caller.
        """
        # Unit conversion must happen before type conversion.  For example,
        # if time is given in float seconds but stored in integer microseconds,
        # the value needs to be scaled by one million before being converted
        # to an integer.

        # Unit conversion
        if units != "":
            value = unit.Converter(units)(value, self.nxunits)
        # Type conversion
        nxtype, nxdims, value = _convert_type(value, self.nxtype)
        # Assignment
        self._value = value
        self.nxtype = nxtype
        self.nxdims = nxdims
        self._dirty = True

    nxdata = property(fget=nxdata_as, fset=nxdata_from,
                      doc="The data in default units")

    def _get_units(self):
        try:
            return self.units.nxdata
        except AttributeError:
            return ""
    def _set_units(self, value):
        try:
            self.units.nxdata = value
        except AttributeError:
            self.units = NXattr(value)
    nxunits = property(_get_units, _set_units, doc="SDS units attribute")

class PylabPlotter(object):
    """
    Matplotlib plotter object for NeXus data nodes.
    """
    def plot(self, signal, axes, title, errorbar=False, hold=False, log=False, **opts):
        """
        Plot the data entry.

        Raises NeXusError if the data cannot be plotted.
        """
        import pylab
        if not hold: pylab.clf()

        # Find the centers of the bins for histogrammed data
        axis_data = centers(signal, axes)

        #One-dimensional Plot
        if len(signal.nxdims) == 1:
            if signal.units.nxdata == 'counts' and errorbar:
                myopts=copy(opts)
                myopts.setdefault('fmt','o')
                myopts.setdefault('linestyle','None')
                pylab.errorbar(axis_data[0], signal.nxdata,
                               numpy.sqrt(signal.nxdata), **myopts)
            else:
                pylab.scatter(axis_data[0], signal.nxdata, **opts)
            if log: pylab.gca().set_yscale('log')
            pylab.xlabel(label(axes[0]))
            pylab.ylabel(label(signal))
            pylab.title(title)

        #Two dimensional plot
        elif len(signal.nxdims) == 2:
            #gridplot = pylab.pcolormesh
            gridplot = imshow_irregular
            if log:
                gridplot(axis_data[1], axis_data[0],
                         numpy.log10(signal.nxdata.T+1), **opts)
            else:
                gridplot(axis_data[1], axis_data[0],
                         signal.nxdata, **opts)
            pylab.xlabel(label(axes[1]))
            pylab.ylabel(label(axes[0]))
            pylab.title(title)

        # No support for higher dimensions yet
        else:
            raise NeXusError, "Cannot plot a dataset of rank 3 or greater."
    @staticmethod
    def show():
        import pylab
        pylab.show()

class NXgroup(NXnode):
    """
    A NeXus group node.

    This is a subclass of NXnode and is the base class for the specific
    NeXus group classes, e.g., NXentry, NXsample, NXdata.

    NXgroup(*items, **opts)

    Parameters
    ----------
    The NXgroup parameters consist of a list of positional and/or keyword
    arguments.

    Positional Arguments : These must be valid NeXus nodes, either an SDS
    or a NeXus group. These are added without modification as children of this
    group.

    Keyword Arguments : Apart from a list of special keywords shown below,
    keyword arguments are used to add children to the group using the keywords
    as attribute names. The values can either be valid SDS data or NXgroups,
    in which case the 'nxname' attribute is changed to the keyword, or they
    can be numerical or string data, which are converted to SDS objects.

    Special Keyword Arguments:

    name : string
        The name of the NXgroup, which is directly accessible as the NXgroup
        attribute 'nxname'. If the NXgroup is initialized as the attribute of
        a parent node, the name is automatically set to the name of this
        attribute. If 'nxclass' is specified and has the usual prefix 'NX',
        the default name is the class name without this prefix.
    nxclass : string
        The class of the NXgroup.
    entries : dict
        A dictionary containing a list of group entries. This is an
        alternative way of adding group entries to the use of keyword
        arguments.
    file : filename
        The file from which the SDS has been read.
    path : string
        The path to this node with respect to the root of the NeXus tree,
        using the convention for unix file paths.
    group : NXnode (NXgroup or subclass of NXgroup)
        The parent NeXus node, which is accessible as the group attribute
        'nxgroup'. If the group is initialized as the attribute of
        a parent node, this is set to the parent node.

    Attributes
    ----------
    nxclass : string
        The class of the NXnode.
    nxname : string
        The name of the SDS.
    nxentries : dict
        A dictionary of all the group entries.
    nxattrs : dict
        A dictionary of the group attributes.
    nxpath : string
        The path to this node with respect to the root of the NeXus tree. For
        NeXus data read from a file, this will be a group of class NXroot, but
        if the NeXus tree was defined interactively, it can be any valid
        NXgroup. This is determined by recursively accessing the 'nxgroup'
        attributes of the parent nodes.

    Methods
    -------
    nxdir(self, attrs=False):
        Print the group directory.

        The directory is a list of NeXus objects within this group, either
        NeXus groups or SDS data. If 'attrs' is True, SDS attributes are
        displayed. If 'recursive' is True, the contents of child groups are
        also displayed.

    nxtree(self, attrs=True):
        Print the SDS and its attributes.

        It invokes the 'nxdir' method with both 'attrs' and 'recursive'
        set to True.

    nxsave(filename, format='w5')
        Save the NeXus group into a file

        The object is wrapped in an NXroot group (with name 'root') and an
        NXentry group (with name 'entry'), if necessary, in order to produce
        a valid NeXus file.

    nxsave(filename, format='w5')
        Save the NeXus group into a file

        The object is wrapped in an NXroot group (with name 'root') and an
        NXentry group (with name 'entry'), if necessary, in order to produce
        a valid NeXus file.

    Examples
    --------
    >>> x = SDS(np.linspace(0,2*np.pi,101), units='degree')
    >>> entry = NXgroup(x, name='entry', nxclass='NXentry')
    >>> entry.sample = NXgroup(temperature=SDS(40.0,units='K'),
                               nxclass='NXsample')
    >>> entry.sample.nxtree()
    sample:NXsample
      temperature = 40.0
        @units = K

    Note: All the currently defined NeXus classes are defined as subclasses
          of the NXgroup class. It is recommended that these are used
          directly, so that the above examples become:

    >>> entry = NXentry(x)
    >>> entry.sample = NXsample(temperature=SDS(40.0,units='K'))

    """
    # Plotter to use for nxplot calls
    _plotter = PylabPlotter()

    def __init__(self, name=None, nxclass=NXnode.nxclass,
                 attrs={},entries={}, group=None):
        self.nxclass = nxclass
        self.nxname = name if name else nxclass[2:]
        self.nxgroup = group
        # Convert children and NeXus attributes to python attributes
        self._setattrs(attrs)
        for k,v in entries.items():
            setattr(self, k, v)
    def _str_value(self,indent=0):
        return ""

    def __getattr__(self, key):
        """
        Provide direct access to nodes via nxclass name.
        """
        if key.startswith('NX'):
            return self.nxcomponent(key)
        raise KeyError(key+" not in "+self.nxclass+":"+self.nxname)

    def NOT__setattr__(self, name, value):
        """
        Set a node attribute as a node or regular Python attribute.

        It is assumed that attributes starting with 'nx' or '_' are regular
        Python attributes. All other attributes are converted to valid NXnodes,
        with class SDS, NXgroup, or a sub-class of NXgroup, depending on the
        assigned value.

        The internal value of the attribute name, i.e., 'nxname', is set to the
        attribute name used in the assignment.  The parent group of the
        attribute, i.e., 'nxgroup', is set to the parent node of the attribute.

        If the assigned value is a numerical (scalar or array) or string object,
        it is converted to an object of class SDS, whose attribute, 'nxdata',
        is set to the assigned value.
        """
        if isinstance(value, NXattr) or name.startswith('nx') or name.startswith('_'):
            object.__setattr__(self, name, value)
        elif isinstance(value, NXnode):
            value.nxgroup = self
            value.nxname = name
            if isinstance(value, NXgroup): value._file = None
            object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, SDS(value=value, name=name, group=self))

    def nxcomponent(self, nxclass):
        """
        Find all child nodes that have a particular class.
        """
        return [E for name,E in self.nxentries.items() if E.nxclass==nxclass]

    def nxsignals(self):
        """
        Return a dictionary of SDS's containing signal data.

        The key is the value of the signal attribute.
        """
        signals = {}
        for node in self.nxentries.values():
            if 'signal' in node.nxattrs:
                signals[node.signal.nxdata] = node
        return signals

    def nxsignal(self):
        """
        SDS containing the signal data.
        """
        for node in self.nxentries.values():
            if 'signal' in node.nxattrs and str(node.signal.nxdata) == '1':
                if 1 in node.nxdims: self._fixaxes(node)
                return self.__dict__[node.nxname]
        return None

    def nxaxes(self):
        """
        List of SDSs containing the axes.
        """
        return [getattr(self, name)
                for name in _parse_axes(self.nxsignal().axes.nxdata)]

    def nxerrors(self):
        """
        SDS containing the signal errors.
        """
        try:
            return self.nxentries['errors']
        except KeyError:
            return None

    def nxplot(self, signal=None, **opts):
        """
        Plot data contained within the group.

        Raises NeXusError if the data could not be plotted.
        """
        group = self

        # Find a plottable signal
        if signal is None:
            signal = group.nxsignal()
            if not signal:
                raise NeXusError('No plottable signal defined')

        # Find errors
        errors= group.nxerrors()

        # Find the associated axes
        axes = group.nxaxes()

        # Construct title
        path = []
        node = group
        title = ''
        while node.nxgroup is not None:
            if node.nxclass == 'NXentry':
                title = node.title.nxdata
            node = node.nxgroup

        # Plot with the available plotter
        self._plotter.plot(signal, axes, title, errors, **opts)

class NXlink(NXnode):
    """
    NeXus linked node.

    The real node will be accessible by following the nxlink attribute.
    """
    def __init__(self,name,nxclass="",attrs={},group=None):
        self.nxclass = nxclass
        self.nxname = name
        self.nxlink = None  # This will be filled at the end of read
        self.nxgroup = None
        # Convert NeXus attributes to python attributes
        self._setattrs(attrs)
        self._link_target = attrs['target'].nxdata
    def __str__(self):
        return "NXlink(%s)"%(self._link_target)
    def _str_tree(self,indent=0,attrs=False,recursive=False):
        return " "*indent+self.nxname+' -> '+self._link_target

class Unknown(NXnode):
    """
    Unknown group type; class does not start with NX or SDS.
    """
    def __init__(self, name="unknown", nxclass="unknown"):
        self.nxname = nxname
        self.nxclass = nxclass

    def __repr__(self):
        return "Unknown('%s','%s')"%(self.nxname,self.nxclass)


def centers(signal, axes):
    """
    Return the centers of the axes regardless if the axes contain
    bin boundaries or centers.
    """
    def findc(axis, dimlen):
        if axis.nxdims[0] == dimlen+1:
            return (axis.nxdata[:-1] + axis.nxdata[1:])/2
        else:
            assert axis.nxdims[0] == dimlen
            return axis.nxdata
    return [findc(a,signal.nxdims[i]) for i,a in enumerate(axes)]

def setmemory(value):
    """
    Set the memory limit for data arrays
    """
    global NX_MAX_MEMORY
    NX_MAX_MEMORY = value

def label(field):
    """
    Construct a label for a data field suitable for use on a graph axis.
    """
    if hasattr(field,'long_name'):
        return field.long_name.nxdata
    elif hasattr(field,'units'):
        return "%s (%s)"%(field.nxname,field.units.nxdata)
    else:
        return field.nxname

def imshow_irregular(x,y,z):
    import pylab
    from matplotlib.ticker import FormatStrFormatter
    ax = pylab.gca()
    im = pylab.mpl.image.NonUniformImage(ax, extent=(x[0],x[-1],y[0],y[-1]), origin=None)
    im.set_data(x,y,z)
    ax.images.append(im)
    ax.set_xlim(x[0],x[-1])
    ax.set_ylim(y[0],y[-1])
    pylab.colorbar(im) #format=FormatStrFormatter('$10^{%d}$')
    pylab.gcf().canvas.draw_idle()

# File level operations
def load(filename, mode='r'):
    """
    Read a NeXus file, returning a tree of nodes
    """
    file = NeXusTree(filename,mode)
    tree = file.readfile()
    file.close()
    return tree

def save(filename, node, format='w5'):
    """
    Write a NeXus file from a tree of nodes. Node is wrapped in an NXentry if necessary.
    """
    if node.nxclass == "NXroot":
        tree = node
    elif node.nxclass == "NXentry":
        tree = NXroot(node)
    else:
        tree = NXroot(NXentry(node))
    file = NeXusTree(filename, format)
    file.writefile(tree)
    file.close()

def tree(file):
    """
    Read and summarize the named nexus file.
    """
    nxfile = load(file)
    nxfile.nxtree()

def demo(argv):
    """
    Process command line commands in argv.  argv should contain
    program name, command, arguments, where command is one of
    the following:
        copy fromfile.nxs tofile.nxs
        ls f1.nxs f2.nxs ...
    """
    if len(argv) > 1:
        op = argv[1]
    else:
        op = 'help'
    if op == 'ls':
        for f in argv[2:]: tree(f)
    elif op == 'copy' and len(argv)==4:
        nxfile = load(argv[2])
        save(argv[3], nxfile)
    elif op == 'plot' and len(argv)==4:
        node = load(argv[2])
        for entry in argv[3].split('.'):
            node = getattr(node,entry)
        node.nxplot()
        node._plotter.show()

    else:
        usage = """
usage: %s cmd [args]
    copy fromfile.nxs tofile.nxs
    ls *.nxs
    plot file.nxs entry.data
        """%(argv[0],)
        print usage

if __name__ == "__main__":
    import sys
    demo(sys.argv)
