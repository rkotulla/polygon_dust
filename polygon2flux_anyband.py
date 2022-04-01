#!/usr/bin/env python3

import os
import sys
import numpy
import matplotlib
import matplotlib.path as mpltPath
import scipy.ndimage

import astropy.io.fits as pyfits
import astropy.table

import astropy.wcs
import pandas
import argparse

# bufferzone = 3

def calzetti(wl):
    micron = numpy.array(wl) / 1e4
    # print("microns", micron)
    # print(1.04 / micron)
    red_part = micron > 0.63
    blue = ~red_part

    # print("red part:", red_part)
    koeff = numpy.zeros_like(micron, dtype=numpy.float)
    # print("raw koeff:", koeff)
    koeff[red_part] = 2.659 * (-1.857 + 1.040 / micron[red_part]) + 4.05
    koeff[blue] = 2.659 * (
                -2.156 + (1.509 / micron[blue]) - 0.198 / micron[blue] ** 2 + 0.011 / micron[blue] ** 3) + 4.05

    # print("final koeff:", koeff)
    return koeff



def measure_polygons(polygon_list, image, wcs, edgewidth=1):

    bufferzone = edgewidth + 2

    iy,ix = numpy.indices(image.shape)
    # print(iy)
    # print(ix)
    # print(ix.ravel())
    index_xy = numpy.hstack((ix.reshape((-1,1)), iy.reshape((-1,1))))
    # print(index_xy)
    # print(index_xy.shape)

    edge_kernel = numpy.ones((2*edgewidth+1, 2*edgewidth+1))

    polygon_data = []
    mask_image = numpy.zeros_like(image)
    edge_image = numpy.zeros_like(image)

    for polygon in polygon_list:

        # sys.stdout.write(".")
        # sys.stdout.flush()

        # first, convert ra/dec to x/y
        xy = wcs.all_world2pix(polygon, 0)
        # print(xy)

        #
        # to speed things up, don't work on the whole image, but
        # rather only on the little area around and including the polygon
        #
        min_xy = numpy.floor(numpy.min(xy, axis=0)).astype(numpy.int) - [bufferzone,bufferzone]
        min_xy[min_xy < 0] = 0
        max_xy = numpy.ceil(numpy.max(xy, axis=0)).astype(numpy.int) + [bufferzone,bufferzone]
        # print(min_xy, max_xy)

        max_x, max_y = max_xy[0], max_xy[1]
        min_x, min_y = min_xy[0], min_xy[1]

        # cutout the area with points in the region
        poly_ix = ix[ min_y:max_y+1, min_x:max_x+1 ]
        poly_iy = iy[ min_y:max_y+1, min_x:max_x+1 ]
        poly_xy = numpy.hstack((poly_ix.reshape((-1,1)), poly_iy.reshape((-1,1))))
        # print(poly_xy.shape)
        # print(poly_xy)

        # use some matplotlib magic to figure out which points are inside the polygon
        path = mpltPath.Path(xy)
        inside2 = path.contains_points(poly_xy)
        inside2d = inside2.reshape(poly_ix.shape)
        # print(inside2d.shape)

        # to get at the border of the polygon, convolve the mask with a small filter
        widened = scipy.ndimage.convolve(inside2d.astype(numpy.int), edge_kernel,
                               mode='constant', cval=0)

        edge_only_pixels = (widened > 0) & (~inside2d)

        image_region = image[ min_y:max_y+1, min_x:max_x+1 ]

        # generate the check images
        mask_image_region = mask_image[ min_y:max_y+1, min_x:max_x+1 ]
        mask_image_region[inside2d] = image_region[inside2d]

        edge_image_region = edge_image[ min_y:max_y+1, min_x:max_x+1 ]
        edge_image_region[edge_only_pixels] += 1

        n_pixels = numpy.sum(inside2)

        # set some default values in case things go wrong down the line
        total_flux = -1
        center_x = -1
        center_y = -1
        edge_mean = edge_median = edge_area = -1

        if (n_pixels >= 1):
            total_flux = numpy.sum(image_region[inside2d])

            # calculate mean position of points inside polygon
            center_x = numpy.mean( poly_ix[inside2d] )
            center_y = numpy.mean( poly_iy[inside2d] )

            edge_mean = numpy.nanmean( image_region[edge_only_pixels] )
            edge_median = numpy.nanmedian( image_region[edge_only_pixels] )
            edge_area = numpy.sum( edge_only_pixels )

        polygon_data.append([n_pixels, total_flux, center_x, center_y, edge_mean, edge_median, edge_area])

        continue

        # do not use this doe, it's slow as hell
        # path = mpltPath.Path(xy)
        # inside2 = path.contains_points(index_xy)
        # inside2d = inside2.reshape(image.shape)

        # mask_image[inside2d] = 1

    polygon_data = numpy.array(polygon_data)

    return (mask_image, edge_image), polygon_data

if __name__ == "__main__":

    cmdline = argparse.ArgumentParser()
    cmdline.add_argument("--region", dest="region_fn", default=None, type=str,
                         help='region filename for initial cutout')
    cmdline.add_argument("--distance", dest="distance", default=0., type=float,
                         help='distance to galaxy in Mpc')
    cmdline.add_argument("--output", dest="output_fn", default="prepped_phot.csv", type=str,
                         help='filename for output file')
    cmdline.add_argument("--reffits", dest="ref_fits", default="prep4gazelle_ref.fits", type=str,
                         help='reference fits file')
    cmdline.add_argument("files", nargs="+",
                         help="list of input filenames")
    args = cmdline.parse_args()

    region_fn = args.region_fn #sys.argv[1]
    # image_fn = sys.argv[2]

    #
    # Now read the region file
    #
    src_polygons = []
    sky_polygons = []
    with open(region_fn, "r") as regfile:
        lines = regfile.readlines()
        # print("Read %d lines" % (len(lines)))

    for line in lines:
        if (not line.startswith("polygon(")):
            # don't do anything
            continue

        coordinates_text = line.split("polygon(")[1].split(")")[0]
        coordinates = coordinates_text.split(",")
        # print(coordinates)

        coordinates2 = [float(c) for c in coordinates]
        # print(coordinates2)

        coordinates_radec = numpy.array(coordinates2).reshape((-1,2))
        # print(coordinates_radec)

        if (line.find("background") > 0):
            # this is a background lines
            # print("BACKGROUND:", line)
            sky_polygons.append(coordinates_radec)
        else:
            # print("not a background")
            src_polygons.append(coordinates_radec)

    print("Found %d source polygons and %d sky polygons" % (
        len(src_polygons), len(sky_polygons)
    ))

    distance_mpc = args.distance
    distance_cm = distance_mpc * 3.086e24

    #
    # Let's run the integration code on all files, one after another
    #
    for image_fn in args.files:

        print("Working on image file %s (regions: %s)" % (image_fn, region_fn))

        #
        # Now lets read the image
        #
        image_hdu = pyfits.open(image_fn)
        # image_hdu.info()

        image_data = image_hdu['SCI'].data
        wcs = astropy.wcs.WCS(image_hdu['SCI'].header)
        # print(wcs)

        sci_hdr = image_hdu['SCI'].header
        photflam = sci_hdr['PHOTFLAM']
        photplam = sci_hdr['PHOTPLAM']
        zp_ab = -2.5*numpy.log10(photflam) - 5*numpy.log10(photplam) - 2.408
        # print("ZP_AB = %f" % (zp_ab))
        # see https://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints

        # get a filtername
        prim_hdr = image_hdu[0].header
        filter1 = prim_hdr['FILTER1'] if 'FILTER1' in prim_hdr else None
        filter2 = prim_hdr['FILTER2'] if 'FILTER2' in prim_hdr else None
        if ((filter1 is None or filter1.startswith("CLEAR")) and filter2 is not None):
            filtername = filter2
        elif (filter1 is not None and (filter2 is None or filter2.startswith("CLEAR"))):
            filtername = filter1
        else:
            filtername = "%s_%s" % (filter1, filter2)
        # print("FILTERNAME:", filtername)

        # get size of a pixel in degrees
        _pixelsize = astropy.wcs.utils.proj_plane_pixel_scales(wcs)
        pixelsize_degrees = _pixelsize[0]
        # print("pixelsize: %f deg = %f arcsec" % (pixelsize_degrees, pixelsize_degrees*3600.))
        # convert pixelsize from degrees into cm
        pixelsize_cm = numpy.tan(numpy.deg2rad(pixelsize_degrees)) * distance_cm
        pixelsize_kpc = numpy.tan(numpy.deg2rad(pixelsize_degrees)) * distance_mpc * 1000.

        # print("integrating sky polygons")
        _, sky_data = measure_polygons(sky_polygons, image_data, wcs)
        # print("integrating source polygons")
        src_images, src_data = measure_polygons(src_polygons, image_data, wcs)

        src_mask, src_edges = src_images
        # pyfits.PrimaryHDU(data=sky_mask).writeto("sky_mask.fits", overwrite=True)
        pyfits.PrimaryHDU(data=src_mask).writeto("src_mask.fits", overwrite=True)
        pyfits.PrimaryHDU(data=src_edges).writeto("src_edges.fits", overwrite=True)

        #
        # Figure out the average sky background level
        #
        median_sky_level = numpy.median( sky_data[:,1]/sky_data[:,0] )
        # print("Median sky = %f" % (median_sky_level))

        # now apply a sky background subtraction for all source polygons
        background_subtracted_src_data = src_data.copy()
        background_subtracted_src_data[:,1] -= background_subtracted_src_data[:,0] * median_sky_level

        background_subtracted_src_data[:,4] -= median_sky_level
        background_subtracted_src_data[:,5] -= median_sky_level

        df = pandas.DataFrame({
            "PolyArea": background_subtracted_src_data[:,0],
            "IntegratedFlux": background_subtracted_src_data[:,1],
            "Mean_X": background_subtracted_src_data[:,2] + 1, # add 1 since fits starts counting at 1
            "Mean_Y": background_subtracted_src_data[:,3] + 1,
            "Edge_Mean": background_subtracted_src_data[:,4],
            "Edge_Median": background_subtracted_src_data[:,5],
            "Edge_Area": background_subtracted_src_data[:,6],
            })
        df.reset_index(inplace=True)

        bad_photometry = df['IntegratedFlux'] <= 0
        
        df['InstrumentalMagnitude'] = -2.5*numpy.log10(df['IntegratedFlux'])
        df['Magnitude_AB'] = df['InstrumentalMagnitude'] + zp_ab

        df['InstrumentalMagnitude'][bad_photometry] = 99.999
        df['Magnitude_AB'][bad_photometry] = 99.999
        
        # # calculate distance from center in pixels
        # df['change_in_x'] = df['Mean_X'] - 2051.3312
        # df['change_in_y'] = df['Mean_Y'] - 2560.5258
        # df['distance_pixels'] = numpy.hypot(df['change_in_x'], df['change_in_y'])
        # # calculate distance from center in kpc
        # df['distance_kpc'] = 0.01536 * df['distance_pixels']
        #
        # add your conversion here
        df['PolyMean'] = df['IntegratedFlux'] / df['PolyArea']

        df['area_cm2'] = df['PolyArea'] * pixelsize_cm**2   # 1.6e39
        df['area_kpc2'] = df['PolyArea'] * pixelsize_kpc**2 #1.82e-4
        df['transmission'] = df['PolyMean'] / df['Edge_Median']
        df['optical_depth'] = -1 * numpy.log(df['transmission'])

        # convert optical depth to A_X (in this filter) -- the factor 1.087 = 2.5 / e
        df['A_X'] = df['optical_depth'] / 1.087

        # convert A_X into A_V, assuming a standard extinction law
        lambda_V = 5555
        lambda_X = sci_hdr['PHOTPLAM']
        dust_koeffs = calzetti([lambda_V, lambda_X])
        df['A_V'] = df['A_X'] * dust_koeffs[1]/dust_koeffs[0]

        # transmission_constant_f435w = 1.3e21
        draine_constant = 5.3e22
        df['number_atoms'] = draine_constant * df['A_V']
        # df['number_atoms'] = transmission_constant_f435w * df['optical_depth']  # for the F435W filter

        mass_per_atom = 2.3e-24 # that's in grams (using some cosmic mix of H, He, metals)
        df['dustmass_grams'] = df['number_atoms'] * mass_per_atom * df['area_cm2']
        df['dustmass_solarmasses'] = df['dustmass_grams'] / 2.e33

        # print(df['dustmass_solarmasses'])

        # convert mean_x/mean_y to ra/dec
        mean_ra_dec = wcs.all_pix2world(df['Mean_X'], df['Mean_Y'], 1.)
        # print(mean_ra_dec)
        df['RA'] = mean_ra_dec[0]
        df['DEC'] = mean_ra_dec[1]

        # df.info()
        df.to_csv(image_fn[:-5]+"_polygonflux.csv")

        # also save as a votable for ds9
        table = astropy.table.Table.from_pandas(df)
        table.write(image_fn[:-5]+"_polygonflux.vot", format='votable', overwrite=True)
        
        # print("\n\nSKY:")
        # print(sky_data)
        # print("\n\nSources:")
        # print(src_data)



        print("done with image %s" % (image_fn))

    print("all done!")
