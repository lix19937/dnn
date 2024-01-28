import re
import warnings
import numpy as np

numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                           (np.dtype('float64'), ('F', 8)),
                           (np.dtype('uint8'), ('U', 1)),
                           (np.dtype('uint16'), ('U', 2)),
                           (np.dtype('uint32'), ('U', 4)),
                           (np.dtype('uint64'), ('U', 8)),
                           (np.dtype('int16'), ('I', 2)),
                           (np.dtype('int32'), ('I', 4)),
                           (np.dtype('int64'), ('I', 8))]
pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)

__all__ = ['PointCloud',
           'point_cloud_to_path',
           'point_cloud_from_path',
        ]

def _build_dtype(metadata):
    """ Build numpy structured array dtype from pcl metadata.

    Note that fields with count > 1 are 'flattened' by creating multiple
    single-count fields.

    *TODO* allow 'proper' multi-count fields.
    """
    # fieldnames = []
    # typenames = []
    field_types = []
    for f, c, t, s in zip(metadata['fields'],
                          metadata['count'],
                          metadata['type'],
                          metadata['size']):
        np_type = pcd_type_to_numpy_type[(t, s)]
        if c == 1:
            field_types.append((f, np_type))
            # typenames.append(np_type)
        else:
            field_types.extend([('%s_%04d' % (f, i), np_type) for i in range(c)])
    dtype = np.dtype(field_types)
    return dtype

def parse_header(lines):
    """ Parse header of PCD files.
    """
    metadata = {}
    for ln in lines:
        if ln.startswith('#') or len(ln) < 2:
            continue
        match = re.match('(\w+)\s+([\w\s\.]+)', ln)
        if not match:
            warnings.warn("warning: can't understand line: %s" % ln)
            continue
        key, value = match.group(1).lower(), match.group(2)
        if key == 'version':
            metadata[key] = value
        elif key in ('fields', 'type'):
            metadata[key] = value.split()
        elif key in ('size', 'count'):
            metadata[key] = list(map(int, value.split()))
        elif key in ('width', 'height', 'points'):
            metadata[key] = int(value)
        elif key == 'viewpoint':
            metadata[key] = list(map(float, value.split()))
        elif key == 'data':
            metadata[key] = value.strip().lower()
        # TODO apparently count is not required?
    # add some reasonable defaults
    if 'count' not in metadata:
        metadata['count'] = [1]*len(metadata['fields'])
    if 'viewpoint' not in metadata:
        metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if 'version' not in metadata:
        metadata['version'] = '.7'
    return metadata

def parse_ascii_pc_data(f, dtype, metadata):
    """ Use numpy to parse ascii pointcloud data.
    """
    return np.loadtxt(f, dtype=dtype, delimiter=' ')

def parse_binary_pc_data(f, dtype, metadata):
    rowstep = metadata['points']*dtype.itemsize
    # for some reason pcl adds empty space at the end of files
    buf = f.read(rowstep)
    return np.fromstring(buf, dtype=dtype)

def point_cloud_from_fileobj(f):
    """ Parse pointcloud coming from file object f
    """
    header = []
    while True:
        ln = f.readline().strip().decode()
        header.append(ln)
        if ln.startswith('DATA'):
            metadata = parse_header(header)
            dtype = _build_dtype(metadata)
            break
    if metadata['data'] == 'ascii':
        pc_data = parse_ascii_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary':
        pc_data = parse_binary_pc_data(f, dtype, metadata)
    else:
        print('DATA field is neither "ascii" or "binary" or\
                "binary_compressed"')
    return PointCloud(metadata, pc_data)

def point_cloud_from_path(fname):
    """ load point cloud in binary format
    """
    with open(fname, 'rb') as f:
        pc = point_cloud_from_fileobj(f)
    return pc

def write_header(metadata, rename_padding=False):
    """ Given metadata as dictionary, return a string header.
    """
    template = """\
    VERSION {version}
    FIELDS {fields}
    SIZE {size}
    TYPE {type}
    COUNT {count}
    WIDTH {width}
    HEIGHT {height}
    VIEWPOINT {viewpoint}
    POINTS {points}
    DATA {data}
    """
    str_metadata = metadata.copy()

    if not rename_padding:
        str_metadata['fields'] = ' '.join(metadata['fields'])
    else:
        new_fields = []
        for f in metadata['fields']:
            if f == '_':
                new_fields.append('padding')
            else:
                new_fields.append(f)
        str_metadata['fields'] = ' '.join(new_fields)
    str_metadata['size'] = ' '.join(map(str, metadata['size']))
    str_metadata['type'] = ' '.join(metadata['type'])
    str_metadata['count'] = ' '.join(map(str, metadata['count']))
    str_metadata['width'] = str(metadata['width'])
    str_metadata['height'] = str(metadata['height'])
    str_metadata['viewpoint'] = ' '.join(map(str, metadata['viewpoint']))
    str_metadata['points'] = str(metadata['points'])
    tmpl = template.format(**str_metadata)
    return tmpl

def build_ascii_fmtstr(pc):
    """ Make a format string for printing to ascii.

    Note %.8f is minimum for rgb.
    """
    fmtstr = []
    for t, cnt in zip(pc.type, pc.count):
        if t == 'F':
            fmtstr.extend(['%.10f']*cnt)
        elif t == 'I':
            fmtstr.extend(['%d']*cnt)
        elif t == 'U':
            fmtstr.extend(['%u']*cnt)
        else:
            raise ValueError("don't know about type %s" % t)
    return fmtstr

def point_cloud_to_fileobj(pc, fileobj):
    """ Write pointcloud as .pcd to fileobj.
    If data_compression is not None it overrides pc.data.
    """
    metadata = pc.get_metadata()

    header = write_header(metadata)
    fileobj.write(header)
    if metadata['data'].lower() == 'ascii':
        fmtstr = build_ascii_fmtstr(pc)
        np.savetxt(fileobj, pc.pc_data, fmt=fmtstr)
    elif metadata['data'].lower() == 'binary':
        fileobj.write(pc.pc_data.tostring('C'))
    else:
        raise ValueError('unknown DATA type')
    # we can't close because if it's stringio buf then we can't get value after

def point_cloud_to_path(pc, fname):
    with open(fname, 'w') as f:
        point_cloud_to_fileobj(pc, f)

import copy
def _metadata_is_consistent(metadata):
    """ Sanity check for metadata. Just some basic checks.
    """
    checks = []
    required = ('version', 'fields', 'size', 'width', 'height', 'points',
                'viewpoint', 'data')
    for f in required:
        if f not in metadata:
            print('%s required' % f)
    checks.append((lambda m: all([k in m for k in required]),
                   'missing field'))
    checks.append((lambda m: len(m['type']) == len(list(m['count'])) ==
                   len(m['fields']),
                   'length of type, count and fields must be equal'))
    checks.append((lambda m: m['height'] > 0,
                   'height must be greater than 0'))
    checks.append((lambda m: m['width'] > 0,
                   'width must be greater than 0'))
    checks.append((lambda m: m['points'] > 0,
                   'points must be greater than 0'))
    checks.append((lambda m: m['data'].lower() in ('ascii', 'binary',
                   'binary_compressed'),
                   'unknown data type:'
                   'should be ascii/binary/binary_compressed'))
    ok = True
    for check, msg in checks:
        if not check(metadata):
            print('error:', msg)
            ok = False
    return ok

class PointCloud(object):
    """ Wrapper for point cloud data.

    The variable members of this class parallel the ones used by
    the PCD metadata (and similar to PCL and ROS PointCloud2 messages),

    ``pc_data`` holds the actual data as a structured numpy array.

    The other relevant metadata variables are:

    - ``version``: Version, usually .7
    - ``fields``: Field names, e.g. ``['x', 'y' 'z']``.
    - ``size.`: Field sizes in bytes, e.g. ``[4, 4, 4]``.
    - ``count``: Counts per field e.g. ``[1, 1, 1]``. NB: Multi-count field
      support is sketchy.
    - ``width``: Number of points, for unstructured point clouds (assumed by
      most operations).
    - ``height``: 1 for unstructured point clouds (again, what we assume most
      of the time.
    - ``viewpoint``: A pose for the viewpoint of the cloud, as
      x y z qw qx qy qz, e.g. ``[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]``.
    - ``points``: Number of points.
    - ``type``: Data type of each field, e.g. ``[F, F, F]``.
    - ``data``: Data storage format. One of ``ascii``, ``binary`` or ``binary_compressed``.

    See `PCL docs <http://pointclouds.org/documentation/tutorials/pcd_file_format.php>`__
    for more information.
    """

    def __init__(self, metadata, pc_data):
        self.metadata_keys = metadata.keys()
        self.__dict__.update(metadata)
        self.pc_data = pc_data
        self.check_sanity()

    def get_metadata(self):
        """ returns copy of metadata """
        metadata = {}
        for k in self.metadata_keys:
            metadata[k] = copy.copy(getattr(self, k))
        return metadata

    def check_sanity(self):
        # pdb.set_trace()
        md = self.get_metadata()
        assert(_metadata_is_consistent(md))
        assert(len(self.pc_data) == self.points)
        assert(self.width*self.height == self.points)
        assert(len(self.fields) == len(self.count))
        assert(len(self.fields) == len(self.type))

    def save(self, fname):
        self.save_pcd(fname, 'binary')

    def save_pcd(self, fname):
        with open(fname, 'w') as f:
            point_cloud_to_fileobj(self, f)

    def save_pcd_to_fileobj(self, fileobj):
        point_cloud_to_fileobj(self, fileobj)

    def copy(self):
        new_pc_data = np.copy(self.pc_data)
        new_metadata = self.get_metadata()
        return PointCloud(new_metadata, new_pc_data)

    @staticmethod
    def from_path(fname):
        return point_cloud_from_path(fname)

    @staticmethod
    def from_fileobj(fileobj):
        return point_cloud_from_fileobj(fileobj)
