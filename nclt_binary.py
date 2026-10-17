"""Fixed-record NCLT Hokuyo reader with a sensor-only malformed-tail audit."""
from pathlib import Path
import numpy as np

HOKUYO_DTYPE=np.dtype([('timestamp','<u8'),('ranges','<u2',(1081,))])


def read_checked_hokuyo(path,odometry_timestamps,imu_timestamps):
    path=Path(path)
    data=np.fromfile(path,dtype=HOKUYO_DTYPE)
    original=len(data);trailing=path.stat().st_size%HOKUYO_DTYPE.itemsize
    lower=min(np.min(odometry_timestamps),np.min(imu_timestamps))-1e6
    upper=max(np.max(odometry_timestamps),np.max(imu_timestamps))+1e6
    bad=np.flatnonzero((data['timestamp']<lower)|(data['timestamp']>upper))
    if len(bad):
        first=int(bad[0])
        if not np.array_equal(bad,np.arange(first,len(data))):
            raise ValueError('Malformed interior Hokuyo timestamps; explicit audit needed')
        data=data[:first]
    if not len(data):raise ValueError('No complete valid Hokuyo records')
    if np.any(np.diff(data['timestamp'].astype(np.int64))<=0):
        raise ValueError('Hokuyo timestamps are not strictly increasing')
    audit=dict(source_file_bytes=path.stat().st_size,record_bytes=HOKUYO_DTYPE.itemsize,
               complete_records_read=original,accepted_records=len(data),
               incomplete_trailing_bytes=trailing,invalid_suffix_records=len(bad),
               timestamp_bounds_source='Provided odometry/IMU sensor timestamps +/- 1s; no ground truth',
               accepted_first_timestamp_us=int(data['timestamp'][0]),
               accepted_last_timestamp_us=int(data['timestamp'][-1]))
    return data,audit
