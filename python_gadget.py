from matplotlib.pyplot import axis, connect
import numpy as np
import torch
import torchkbnufft as tkbn
from gadgetron.examples.recon_buffers import *
import ismrmrd

# import sigpy as sp
# import sigpy.mri as mr

import time
import logging

GA = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SPOKES = 16
SLIDING_WINDOW = 8

def GA_traj(start_index, nspokes, spokelength):
    # Generate the Golden-Angle trajectory following the torchkbnufft's examples
    angles = -GA*torch.arange(start_index, start_index+nspokes, dtype=torch.float32, device=device).unsqueeze_(1)
    pos = torch.linspace(-np.pi, np.pi, spokelength, device=device).unsqueeze_(0)
    kx = torch.mm(torch.cos(angles), pos)
    ky = torch.mm(torch.sin(angles), pos)
    return torch.stack((kx.flatten(), ky.flatten()))


def get_first_index_of_non_empty_header(header):
    # if the data is undersampled, the corresponding acquisitonHeader will be filled with 0 
    # in order to catch valuable information, we need to catch an non-empty header
    # using the following lines 
      
    print(np.shape(header))
    dims=np.shape(header)

    idx = []
    slices = list(set([header[i].idx.slice for i in range(dims[0])]))
    for ii in range(0,dims[0]):
        for i in slices:
            if (header[ii].scan_counter > 0) and (i == header[ii].idx.slice):
                idx.append(ii)
                break
        if len(idx) == len(slices):
            break
    return idx


def send_reconstructed_images(connection, data_array, acq_header):
    # the function creates an new ImageHeader for each 4D dataset [RO,E1,E2,CHA]
    # copy information from the acquisitonHeader
    # fill additionnal fields
    # and send the reconstructed image and ImageHeader to the next gadget
    # some field are not correctly filled like image_type that floattofix point doesn't recognize , why ?
    
    dims=data_array.shape     
    print(dims)
    base_header=ismrmrd.ImageHeader()
    base_header.version=2
    ndims_image=(dims[0], dims[1], dims[2], dims[3])
    base_header.channels = ndims_image[3]       
    base_header.matrix_size = (data_array.shape[0],data_array.shape[1],data_array.shape[2])
  
    I=np.zeros((dims[0], dims[1], dims[2], dims[3]))
    for slc in range(0, dims[6]):
        for n in range(0, dims[5]):
            for s in range(0, dims[4]):
                I=data_array[:,:,:,:,s,n,slc]
                # I = (I-I.min())/(I.max()-I.min())
                base_header.position = acq_header[slc].position
                base_header.read_dir = acq_header[slc].read_dir
                base_header.phase_dir = acq_header[slc].phase_dir
                base_header.slice_dir = acq_header[slc].slice_dir
                base_header.patient_table_position = acq_header[slc].patient_table_position
                base_header.acquisition_time_stamp = acq_header[slc].acquisition_time_stamp
                base_header.image_index = 0 
                base_header.image_series_index = 0
                base_header.data_type = ismrmrd.DATATYPE_USHORT
                base_header.image_type= ismrmrd.IMTYPE_MAGNITUDE
                base_header.repetition=acq_header[slc].idx.repetition
                base_header.slice=slc              
                image_array= ismrmrd.image.Image.from_array(I, headers=base_header)
                connection.send(image_array)


def GreLocPythonGadget(connection):
    '''logging.info("Python reconstruction running - reconstructing images from acquisition buffers.")

    start = time.time()


    for data in connection:
        images = reconstruct(data, connection.header)
        np.save('/opt/data/reconstructed.npy', images)
        send_images(connection, images)

    logging.info(f"Python reconstruction done. Duration: {(time.time() - start):.2f} s")'''

    for acquisition in connection:
      
        #acquisition is a vector of structure called reconBit
        #print(type(acquisition[0]))

        for reconBit in acquisition:

            # print(type(reconBit))
            # # reconBit.ref is the calibration for parallel imaging
            # # reconBit.data is the undersampled dataset
            # print('-----------------------')
	        # # each of them include a specific header and the kspace data
            # print(type(reconBit.data.headers))
            # print(type(reconBit.data.data))

            # print(reconBit.data.headers.shape)
            # print(reconBit.data.data.shape)
            
            index= get_first_index_of_non_empty_header(reconBit.data.headers.flat)
            # repetition=reconBit.data.headers.flat[index].idx.repetition 
            # print(repetition)
            reference_header=reconBit.data.headers.flat[index]

            np.save('/opt/data/header.npy', reconBit.data.headers)
            np.save('/opt/data/origin.npy', reconBit.data.data)
            
            # 2D ifft
            dims=reconBit.data.data.shape
            dims_rss = list(dims)
            dims_rss[3] = 1
            im_rss = np.zeros(dims_rss, reconBit.data.data.dtype)
            for slc in range(0, dims[6]):
                for s in range(0, dims[5]):
                    for n in range(0, dims[4]):
                        #catch 4D dataset [RO E1 E2 CHA] from [RO E1 E2 CHA N S SLC]
                        kspace=reconBit.data.data[:,:,:,:,n,s,slc]
                        # tranpose from [RO E1 E2 CHA] to [CHA E2 E1 RO]
                        ksp=np.transpose(kspace, (3, 2 , 1, 0))
                        # 2D ifft in bart 
                        F = sp.linop.FFT(ksp.shape, axes=(-1, -2))
                        I = F.H * ksp
                        
		                #I is a 4D dataset, put back the data into 7D ndarray
                        #im is a 7D dataset
                        I_ = np.sum(np.abs(np.transpose(I, (3, 2 , 1, 0)) )**2, axis=3)**0.5
                        # I_ = (I_-I_.min())/(I_.max()-I_.min())
                        im_rss[:,:,:,0,n,s,slc] = I_

            np.save('/opt/data/reconstructed.npy', np.mean(im_rss, axis=-2, keepdims=True).astype(float))
            send_reconstructed_images(connection, np.mean(im_rss*1e7, axis=-2, keepdims=True).astype(np.uint16), reference_header)


def accumulate_GAspokes(acquisitions, spokes_num=SPOKES, sliding_window=SLIDING_WINDOW):
    accumulated_acquisitions = []

    def assemble_buffer(acqs):
        logging.info(f"Assembling buffer from {len(acqs)} acquisitions.")
        number_of_channels = acqs[0].data.shape[0]
        number_of_samples = acqs[0].data.shape[1]

        buffer = np.zeros(
            (number_of_channels,
             spokes_num*number_of_samples),
            dtype=np.complex64
        )

        for ind, acq in enumerate(acqs):
            buffer[:, ind*number_of_samples:(ind+1)*number_of_samples] = acq.data

        return buffer

    for acquisition in acquisitions:
        # print(np.shape(acquisition.data), acquisition.data.dtype)
        accumulated_acquisitions.append(acquisition)
        if not acquisition.idx.slice == 0: return
        elif len(accumulated_acquisitions) == spokes_num:
            yield acquisition, assemble_buffer(accumulated_acquisitions)
            del accumulated_acquisitions[:sliding_window]


def reconstruct_images(buffers, header):
    im_size = (header.encoding[0].encodedSpace.matrixSize.x, header.encoding[0].encodedSpace.matrixSize.y)
    adjnufft_ob = tkbn.KbNufftAdjoint(
        im_size=im_size,
        device=device
    )

    def NUFFT_reco(kspace_data, start_index):
        t0 = time.time()
        ktraj = GA_traj(start_index, SPOKES, kspace_data.shape[1]//SPOKES)
        # t1 = time.time()
        # logging.debug("Generate traj within {} s".format(t1-t0))
        dcomp = tkbn.calc_density_compensation_function(ktraj=ktraj, im_size=im_size, num_iterations=10)
        # t2 = time.time()
        # logging.debug("Generate dcomp within {} s".format(t2-t1))
        output = adjnufft_ob(torch.tensor(kspace_data, device=device).unsqueeze_(0) * dcomp, ktraj).squeeze()
        # output = adjnufft_ob(torch.tensor(kspace_data, device=device).unsqueeze_(0), ktraj).squeeze()
        # logging.debug("ANUFFT within {} s".format(time.time() - t2))
        return output

    def combine_channels(image_data):
        # The buffer contains complex images, one for each channel. We combine these into a single image
        # through a sum of squares along the channels (axis 0).
        output = torch.sqrt(torch.sum(torch.square(torch.abs(image_data)), axis=0)).cpu().numpy()
        return output

    for reference, data in buffers:
        t1 = time.time()
        logging.info("Working on Buffer {}".format((reference.idx.kspace_encode_step_1+1)//SLIDING_WINDOW-1))
        yield ismrmrd.image.Image.from_array(
            combine_channels(NUFFT_reco(data, reference.idx.kspace_encode_step_1-SPOKES+1)),
            acquisition=reference
        )
        t2 = time.time()
        logging.info("End Buffer {} in {:.2f} s".format((reference.idx.kspace_encode_step_1 + 1) // SLIDING_WINDOW - 1, (t2 - t1)))


def GoldenAnglePythonGadget(connection):
    start = time.time()
    connection.filter(ismrmrd.Acquisition)
    acquisitions = iter(connection)
    buffers = accumulate_GAspokes(acquisitions)
    images = reconstruct_images(buffers, connection.header)
    for image in images:
        connection.send(image)
    end = time.time()
    logging.info('Total finish within {:.2f} s'.format(end-start))
    # np.save('/opt/data/buffers.npy', buffers)
    '''for i, acquisition in enumerate(connection):
        print(i)
        print(np.shape(acquisition.data), acquisition.data.dtype)
        print(np.shape(acquisition.traj), acquisition.traj.dtype)
        # np.save('/opt/data/GAtest{}.npy'.format(i), acquisition)'''