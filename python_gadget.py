from matplotlib.pyplot import axis, connect
import numpy as np
import torch
import torchkbnufft as tkbn
import ismrmrd

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