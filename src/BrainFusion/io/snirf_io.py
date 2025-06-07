"""
SNIRF File Utilities

Provides functions for creating and reading SNIRF files in HDF5 format.
Compatible with Sphinx documentation generator.
"""

import h5py
import numpy as np


def create_snirf_file(filename, format_version='1.0', metadata=None, data_time_series=None,
                      time_points=None, measurement_lists=None,
                      source_pos_3d=None, detector_pos_3d=None, wavelengths=None,
                      wavelengths_emission=None, source_pos_2d=None, detector_pos_2d=None,
                      frequencies=None, time_delays=None, time_delay_widths=None,
                      moment_orders=None, correlation_time_delays=None,
                      correlation_time_delay_widths=None, source_labels=None,
                      detector_labels=None, landmark_pos_2d=None, landmark_pos_3d=None,
                      landmark_labels=None, coordinate_system='',
                      coordinate_system_description='', use_local_index=0,
                      stim_lists=None):
    """
    Create a SNIRF-compliant HDF5 file

    :param filename: Output HDF5 file path
    :type filename: str
    :param format_version: SNIRF format version
    :type format_version: str, optional
    :param metadata: Dictionary of metadata tags
    :type metadata: dict, optional
    :param data_time_series: Time-series data array
    :type data_time_series: numpy.ndarray
    :param time_points: Time vector corresponding to data
    :type time_points: numpy.ndarray
    :param measurement_lists: List of measurement list dictionaries
    :type measurement_lists: list, optional
    :param source_pos_3d: 3D positions of light sources
    :type source_pos_3d: numpy.ndarray, optional
    :param detector_pos_3d: 3D positions of detectors
    :type detector_pos_3d: numpy.ndarray, optional
    :param wavelengths: Light wavelengths for measurement
    :type wavelengths: numpy.ndarray, optional
    :param wavelengths_emission: Emission wavelengths
    :type wavelengths_emission: numpy.ndarray, optional
    :param source_pos_2d: 2D positions of light sources
    :type source_pos_2d: numpy.ndarray, optional
    :param detector_pos_2d: 2D positions of detectors
    :type detector_pos_2d: numpy.ndarray, optional
    :param frequencies: Frequency modulation values
    :type frequencies: numpy.ndarray, optional
    :param time_delays: Time delay values
    :type time_delays: numpy.ndarray, optional
    :param time_delay_widths: Time delay distribution widths
    :type time_delay_widths: numpy.ndarray, optional
    :param moment_orders: Moment orders for analysis
    :type moment_orders: numpy.ndarray, optional
    :param correlation_time_delays: Correlation time delays
    :type correlation_time_delays: numpy.ndarray, optional
    :param correlation_time_delay_widths: Correlation time delay widths
    :type correlation_time_delay_widths: numpy.ndarray, optional
    :param source_labels: Source identification labels
    :type source_labels: list, optional
    :param detector_labels: Detector identification labels
    :type detector_labels: list, optional
    :param landmark_pos_2d: 2D landmark positions
    :type landmark_pos_2d: numpy.ndarray, optional
    :param landmark_pos_3d: 3D landmark positions
    :type landmark_pos_3d: numpy.ndarray, optional
    :param landmark_labels: Landmark identification labels
    :type landmark_labels: list, optional
    :param coordinate_system: Coordinate system identifier
    :type coordinate_system: str, optional
    :param coordinate_system_description: Coordinate system description
    :type coordinate_system_description: str, optional
    :param use_local_index: Flag for using local indexing (0 or 1)
    :type use_local_index: int, optional
    :param stim_lists: List of stimulation dictionaries
    :type stim_lists: list, optional
    """
    default_metadata = {
        'SubjectID': 'Unknown',
        'MeasurementDate': '2024-09-18',
        'MeasurementTime': '15:30:00',
        'LengthUnit': 'cm',
        'TimeUnit': 's',
        'FrequencyUnit': 'Hz'
    }

    # Create variable-length string datatype
    varlen_str_dtype = h5py.string_dtype(encoding='ascii', length=None)

    with h5py.File(filename, 'w') as f:
        # Set format version
        f.create_dataset('formatVersion', dtype=varlen_str_dtype, data=format_version)
        nirs_group = f.create_group('/nirs')

        # Create metadata tags
        metadata_group = nirs_group.create_group('metaDataTags')
        if metadata:
            for key, value in metadata.items():
                metadata_group.create_dataset(key, dtype=varlen_str_dtype, data=value)
        else:
            for key, value in default_metadata.items():
                metadata_group.create_dataset(key, dtype=varlen_str_dtype, data=value)

        # Create primary data container
        data_group = nirs_group.create_group('data1')
        if data_time_series is not None:
            data_group.create_dataset('dataTimeSeries', dtype='f8', data=data_time_series)
        if time_points is not None:
            data_group.create_dataset('time', dtype='f8', data=time_points)

        # Create measurement lists
        if measurement_lists:
            i = 1
            for measurement in measurement_lists:
                measurement_list_group = data_group.create_group(f'measurementList{i}')

                # Required fields
                measurement_list_group.create_dataset('sourceIndex', dtype='i4',
                                                      data=measurement.get('sourceIndex', 1))
                measurement_list_group.create_dataset('detectorIndex', dtype='i4',
                                                      data=measurement.get('detectorIndex', 1))
                measurement_list_group.create_dataset('wavelengthIndex', dtype='i4',
                                                      data=measurement.get('wavelengthIndex', 1))
                measurement_list_group.create_dataset('dataType', dtype='i4',
                                                      data=measurement.get('dataType', 1))
                measurement_list_group.create_dataset('dataTypeIndex', dtype='i4',
                                                      data=measurement.get('dataTypeIndex', 1))

                # Optional fields
                measurement_list_group.create_dataset('wavelengthActual', dtype='f8',
                                                      data=measurement.get('wavelengthActual', np.nan))
                measurement_list_group.create_dataset('wavelengthEmissionActual', dtype='f8',
                                                      data=measurement.get('wavelengthEmissionActual', np.nan))
                measurement_list_group.create_dataset('dataUnit', dtype=varlen_str_dtype,
                                                      data=measurement.get('dataUnit', ''))
                measurement_list_group.create_dataset('dataTypeLabel', dtype=varlen_str_dtype,
                                                      data=measurement.get('dataTypeLabel', ''))
                measurement_list_group.create_dataset('sourcePower', dtype='f8',
                                                      data=measurement.get('sourcePower', np.nan))
                measurement_list_group.create_dataset('detectorGain', dtype='f8',
                                                      data=measurement.get('detectorGain', np.nan))
                measurement_list_group.create_dataset('moduleIndex', dtype='i4',
                                                      data=measurement.get('moduleIndex', 0))
                measurement_list_group.create_dataset('sourceModuleIndex', dtype='i4',
                                                      data=measurement.get('sourceModuleIndex', 0))
                measurement_list_group.create_dataset('detectorModuleIndex', dtype='i4',
                                                      data=measurement.get('detectorModuleIndex', 0))
                i += 1

        # Create probe section
        probe_group = nirs_group.create_group('probe')

        # Positional data
        if source_pos_3d is not None:
            probe_group.create_dataset('sourcePos3D', dtype='f8', data=source_pos_3d)
        if detector_pos_3d is not None:
            probe_group.create_dataset('detectorPos3D', dtype='f8', data=detector_pos_3d)
        if source_pos_2d is not None:
            probe_group.create_dataset('sourcePos2D', dtype='f8', data=source_pos_2d)
        if detector_pos_2d is not None:
            probe_group.create_dataset('detectorPos2D', dtype='f8', data=detector_pos_2d)
        if landmark_pos_2d is not None:
            probe_group.create_dataset('landmarkPos2D', dtype='f8', data=landmark_pos_2d)
        if landmark_pos_3d is not None:
            probe_group.create_dataset('landmarkPos3D', dtype='f8', data=landmark_pos_3d)

        # Wavelength data
        if wavelengths is not None:
            probe_group.create_dataset('wavelengths', dtype='f8', data=wavelengths)
        if wavelengths_emission is not None:
            probe_group.create_dataset('wavelengthsEmission', dtype='f8', data=wavelengths_emission)

        # Advanced optical parameters
        if frequencies is not None:
            probe_group.create_dataset('frequencies', dtype='f8', data=frequencies)
        if time_delays is not None:
            probe_group.create_dataset('timeDelays', dtype='f8', data=time_delays)
        if time_delay_widths is not None:
            probe_group.create_dataset('timeDelayWidths', dtype='f8', data=time_delay_widths)
        if moment_orders is not None:
            probe_group.create_dataset('momentOrders', dtype='f8', data=moment_orders)
        if correlation_time_delays is not None:
            probe_group.create_dataset('correlationTimeDelays', dtype='f8', data=correlation_time_delays)
        if correlation_time_delay_widths is not None:
            probe_group.create_dataset('correlationTimeDelayWidths', dtype='f8',
                                       data=correlation_time_delay_widths)

        # Label information
        if source_labels is not None:
            probe_group.create_dataset('sourceLabels', dtype=varlen_str_dtype, data=source_labels)
        if detector_labels is not None:
            probe_group.create_dataset('detectorLabels', dtype=varlen_str_dtype, data=detector_labels)
        if landmark_labels is not None:
            probe_group.create_dataset('landmarkLabels', dtype=varlen_str_dtype,
                                       data=landmark_labels)

        # Coordinate system
        probe_group.create_dataset('coordinateSystem', dtype=varlen_str_dtype,
                                   data=coordinate_system)
        probe_group.create_dataset('coordinateSystemDescription', dtype=varlen_str_dtype,
                                   data=coordinate_system_description)
        probe_group.create_dataset('useLocalIndex', dtype='i4', data=use_local_index)

        # Create stimulation section
        if stim_lists:
            for i, stim_dict in enumerate(stim_lists, start=1):
                stim_group = nirs_group.create_group(f'stim{i}')
                if 'name' in stim_dict:
                    stim_group.create_dataset('name', dtype=varlen_str_dtype, data=stim_dict['name'])
                if 'data' in stim_dict and stim_dict['data']:
                    stim_group.create_dataset('data', dtype='f8', data=np.array(stim_dict['data']))
                if 'dataLabels' in stim_dict and stim_dict['dataLabels']:
                    stim_group.create_dataset('dataLabels', dtype=varlen_str_dtype,
                                              data=np.array(stim_dict['dataLabels']).astype('O'))

    print("SNIRF file created successfully")


def read_snirf_file(filename):
    """
    Read and parse a SNIRF file

    :param filename: Path to SNIRF file
    :type filename: str
    :return: Structured dictionary containing file contents
    :rtype: dict
    """
    result = {}

    with h5py.File(filename, 'r') as f:
        # Read format version
        result['formatVersion'] = f['formatVersion'][()].decode('ascii')

        # Read metadata section
        result['metaDataTags'] = {}
        meta_tags = f['/nirs/metaDataTags']
        for key in meta_tags.keys():
            result['metaDataTags'][key] = meta_tags[key][()].decode('ascii')

        # Read data section
        data_group = f['/nirs/data1']
        result['data'] = {
            'dataTimeSeries': data_group['dataTimeSeries'][()],
            'time': data_group['time'][()]
        }

        # Read measurement lists
        result['measurementList'] = []
        measurement_counter = 1
        while f'/nirs/data1/measurementList{measurement_counter}' in data_group:
            ml_group = data_group[f'measurementList{measurement_counter}']

            measurement = {
                'sourceIndex': ml_group['sourceIndex'][()],
                'detectorIndex': ml_group['detectorIndex'][()],
                'wavelengthIndex': ml_group['wavelengthIndex'][()],
                'dataType': ml_group['dataType'][()],
                'dataTypeIndex': ml_group['dataTypeIndex'][()]
            }

            # Optional fields with defaults
            optional_fields = [
                'wavelengthActual', 'wavelengthEmissionActual', 'dataUnit',
                'dataTypeLabel', 'sourcePower', 'detectorGain', 'moduleIndex',
                'sourceModuleIndex', 'detectorModuleIndex'
            ]

            for field in optional_fields:
                if field in ml_group:
                    value = ml_group[field][()]
                    if field in ['dataUnit', 'dataTypeLabel']:
                        value = value.decode('ascii') if value != '' else value
                    measurement[field] = value
                else:
                    measurement[field] = None

            result['measurementList'].append(measurement)
            measurement_counter += 1

        # Read probe section
        probe_group = f['/nirs/probe']
        result['probe'] = {}

        # Positional data
        position_fields = [
            'sourcePos3D', 'detectorPos3D', 'sourcePos2D', 'detectorPos2D',
            'landmarkPos2D', 'landmarkPos3D'
        ]

        for field in position_fields:
            if field in probe_group:
                result['probe'][field] = probe_group[field][()]
            else:
                result['probe'][field] = None

        # Wavelength data
        wavelength_fields = [
            'wavelengths', 'wavelengthsEmission', 'frequencies',
            'timeDelays', 'timeDelayWidths', 'momentOrders',
            'correlationTimeDelays', 'correlationTimeDelayWidths'
        ]

        for field in wavelength_fields:
            if field in probe_group:
                result['probe'][field] = probe_group[field][()]
            else:
                result['probe'][field] = None

        # Label information
        label_fields = [
            'sourceLabels', 'detectorLabels', 'landmarkLabels'
        ]

        for field in label_fields:
            if field in probe_group:
                labels = probe_group[field][()]
                result['probe'][field] = [label.decode('ascii') for label in labels]
            else:
                result['probe'][field] = None

        # Coordinate system
        system_fields = [
            'coordinateSystem', 'coordinateSystemDescription'
        ]

        for field in system_fields:
            if field in probe_group:
                result['probe'][field] = probe_group[field][()].decode('ascii')
            else:
                result['probe'][field] = None

        # Local index flag
        if 'useLocalIndex' in probe_group:
            result['probe']['useLocalIndex'] = probe_group['useLocalIndex'][()]
        else:
            result['probe']['useLocalIndex'] = None

        # Read stimulation data
        result['stimulation'] = []
        stimulation_counter = 1
        while f'/nirs/stim{stimulation_counter}' in f['/nirs']:
            stim_group = f['/nirs'][f'stim{stimulation_counter}']
            stim_dict = {}

            if 'name' in stim_group:
                stim_dict['name'] = stim_group['name'][()].decode('ascii')

            if 'data' in stim_group:
                stim_dict['data'] = stim_group['data'][()]

            if 'dataLabels' in stim_group:
                labels = stim_group['dataLabels'][()]
                stim_dict['dataLabels'] = [label.decode('ascii') for label in labels]

            result['stimulation'].append(stim_dict)
            stimulation_counter += 1

    return result


if __name__ == '__main__':
    # Example usage
    sample_file = "example.snirf"

    # Create sample SNIRF file
    create_snirf_file(
        filename=sample_file,
        data_time_series=np.random.rand(100, 10),
        time_points=np.linspace(0, 10, 100),
        wavelengths=np.array([690, 830]),
        source_labels=["S1", "S2"],
        detector_labels=["D1", "D2"]
    )

    # Read created file
    file_contents = read_snirf_file(sample_file)
    print("File contents:")
    for key in file_contents:
        print(f" - {key} section present")