#include "vtl.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdio>
#include <stdint.h>
#include <string>
#include <vector>
#include <iostream>
#include <map>
#include <cmath>
#include <cassert>
#include "swapbytes.h"
#include "vtlSPoints.h"

using namespace vtl;

#ifdef USE_ZLIB
#include <zlib.h>
#else
#define Z_OK 0
#define uLong size_t
#define uLongf size_t
#endif

const int __one__ = 1;
const bool isCpuLittleEndian = 1 == *(char *)(&__one__); // CPU endianness


// converts zlib status to a human-readable string

std::string zlibstatus(int status)
{
#ifdef USE_ZLIB
    switch (status)
    {
    case Z_OK:
        return "Z_OK";
    case Z_BUF_ERROR:
        return "Z_BUF_ERROR";
    case Z_MEM_ERROR:
        return "Z_MEM_ERROR";
    case Z_STREAM_ERROR:
        return "Z_STREAM_ERROR";
    default:
        std::stringstream str;
        str << "Unknown (" << status << ")";
        return str.str();
    }
#else
    return "zlib missing";
#endif
}

// sends a vector of "doubles" in binary XML/format into filestream f
//  f: destination filestream
//  pos: the vector to be sent
//  usez: true if zlib should be used

size_t write_vectorXML(std::ofstream &f, std::vector<double> const &pos, bool usez, int Nx, int Ny, int Nz, int thermal)
{
    size_t written = 0;

    // convert doubles to floats
    std::vector<float> buffer(pos.size());
    if(thermal == 0){
    	for (int i = 0; i < pos.size(); ++i){
        	buffer[i] = (float)pos[i];
    	}
    }
    else{
	for (int i = 0; i < Nx; ++i){
		for(int j=0;j<Ny;++j){
    			for(int k=0;k<Nz;++k){
    				buffer[i+j*Nx+k*Nx*Ny] = (float) pos[j+k*Ny+i*Ny*Nz];
    			}
    		}
    	}
    }


    if (!usez)
    {
        // data block size
        uint32_t sz = (uint32_t)pos.size() * sizeof(float);
        f.write((char *)&sz, sizeof(uint32_t));
        written += sizeof(uint32_t);
        // data
        f.write((char *)&buffer[0], sz);
        written += sz;
    }
    else
    {
        uLong sourcelen = (uLong)pos.size() * sizeof(float);
        uLongf destlen = uLongf(sourcelen * 1.001) + 12; // see doc
        char *destbuffer = new char[destlen];
#ifdef USE_ZLIB
        int status = compress2((Bytef *)destbuffer, &destlen,
                               (Bytef *)&(buffer[0]), sourcelen, Z_DEFAULT_COMPRESSION);
#else
        int status = Z_OK + 1;
#endif
        if (status != Z_OK)
        {
            std::cout << "ERROR: zlib Error status=" << zlibstatus(status) << "\n";
        }
        else
        {
            //std::cout << "block of size " << sourcelen << " compressed to " << destlen << '\n';
            // blocks description
            uint32_t nblocks = 1;
            f.write((char *)&nblocks, sizeof(uint32_t));
            written += sizeof(uint32_t);
            uint32_t srclen = (uint32_t)sourcelen;
            f.write((char *)&srclen, sizeof(uint32_t));
            written += sizeof(uint32_t);
            uint32_t lastblocklen = 0;
            f.write((char *)&lastblocklen, sizeof(uint32_t));
            written += sizeof(uint32_t);
            uint32_t szblocki = (uint32_t)destlen;
            f.write((char *)&szblocki, sizeof(uint32_t));
            written += sizeof(uint32_t);
            // data
            f.write(destbuffer, destlen);
            written += destlen;
        }

        delete[] destbuffer;
    }

    return written;
}



// export results to paraview (VTK polydata - XML fomat)
//   filename: file name without vtk extension
//   pos:     positions (vector of size 3*number of particles)
//   step:    time step number
//   scalars: scalar fields defined on particles (map linking [field name] <=> [vector of results v1, v2, v3, v4, ...]
//   vectors: vector fields defined on particles (map linking [field name] <=> [vector of results v1x, v1y, v1z, v2x, v2y, ...]

// see http://www.vtk.org/Wiki/VTK_XML_Formats

VTL_API void vtl::export_spoints_XML(std::string const &filename,
                                int step,
                                SPoints const &grid, SPoints const &mygrid,
                                Zip zip, int Nx, int Ny, int Nz, int thermal)
{
#if !defined(USE_ZLIB)
    if (zip==ZIPPED)
    {
        std::cout << "INFO: zlib not present - vtk file will not be compressed!\n";
        zip = UNZIPPED;
    }
#endif

    // build file name (+rankno) + stepno + vtk extension
    std::stringstream s;
    s << filename;
    if (mygrid.id >= 0)
        s << "_r" << mygrid.id;
    s << '_' << std::setw(8) << std::setfill('0') << step << ".vti";
    std::stringstream s2;
    s2 << filename;
    if (mygrid.id >= 0)
        s2 << "r_" << mygrid.id;
    s2 << '_' << std::setw(8) << std::setfill('0') << step << ".vti.tmp";

    // open file
    std::cout << "writing results to " << s.str() << '\n';
    std::ofstream f(s.str().c_str(), std::ios::binary | std::ios::out);
    std::ofstream f2(s2.str().c_str(), std::ios::binary | std::ios::out); // temp binary file
    f << std::scientific;

    size_t offset = 0;
    // header
    f << "<?xml version=\"1.0\"?>\n";

    f << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"";
    f << (isCpuLittleEndian ? "LittleEndian" : "BigEndian") << "\" ";
    f << "header_type=\"UInt32\" "; // UInt64 could be better (?)
    if (zip==ZIPPED)
        f << "compressor=\"vtkZLibDataCompressor\" ";
    f << ">\n";

    f << "  <ImageData ";
    f << "WholeExtent=\""
      << grid.np1[0] << ' ' << grid.np2[0]+1 << ' '
      << grid.np1[1] << ' ' << grid.np2[1]+1 << ' '
      << grid.np1[2] << ' ' << grid.np2[2]+1 << "\" ";
    f << "Origin=\"" << grid.o[0] << ' ' << grid.o[1] << ' ' << grid.o[2] << "\" ";
    f << "Spacing=\"" << grid.dx[0] << ' ' << grid.dx[1] << ' ' << grid.dx[2] << "\">\n";

    f << "    <Piece ";
    f << "Extent=\""
      << mygrid.np1[0] << ' ' << mygrid.np2[0]+1 << ' '
      << mygrid.np1[1] << ' ' << mygrid.np2[1]+1 << ' '
      << mygrid.np1[2] << ' ' << mygrid.np2[2]+1 << "\">\n";

    // ------------------------------------------------------------------------------------
    f << "      <CellData>\n";

    // scalar fields
    for (auto it = mygrid.scalars.begin(); it != mygrid.scalars.end(); ++it)
    {
        //assert(it->second->size() == nbp); // TODO
        f << "        <DataArray type=\"Float32\" ";
        f << " Name=\"" << it->first << "\" ";
        f << " format=\"appended\" ";
        f << " RangeMin=\"0\" ";
        f << " RangeMax=\"1\" ";
        f << " offset=\"" << offset << "\" />\n";
        offset += write_vectorXML(f2, *it->second, (zip==ZIPPED), Nx, Ny, Nz, thermal);
    }

    // vector fields
    for (auto it = mygrid.vectors.begin(); it != mygrid.vectors.end(); ++it)
    {
        //assert(it->second->size() == 3 * nbp); // TODO
        f << "        <DataArray type=\"Float32\" ";
        f << " Name=\"" << it->first << "\" ";
        f << " NumberOfComponents=\"3\" ";
        f << " format=\"appended\" ";
        f << " RangeMin=\"0\" ";
        f << " RangeMax=\"1\" ";
        f << " offset=\"" << offset << "\" />\n";
        offset += write_vectorXML(f2, *it->second, (zip==ZIPPED), Nx, Ny, Nz, thermal);
    }
    f << "      </CellData>\n";

    // ------------------------------------------------------------------------------------
    f << "      <PointData>\n";
    f << "      </PointData>\n";

    f2.close();

    // ------------------------------------------------------------------------------------
    f << "    </Piece>\n";
    f << "  </ImageData>\n";
    // ------------------------------------------------------------------------------------
    f << "  <AppendedData encoding=\"raw\">\n";
    f << "    _";

    // copy temp binary file as "appended" data
    std::ifstream f3(s2.str().c_str(), std::ios::binary | std::ios::in);
    f << f3.rdbuf();
    f3.close();
    // remove temp file
    std::remove(s2.str().c_str());

    f << "  </AppendedData>\n";
    f << "</VTKFile>\n";

    f.close();
}

VTL_API void vtl::export_spoints_XMLP(std::string const &filename,
                                 int step,
                                 SPoints const &grid,
                                 SPoints const &mygrid,
                                 std::vector<SPoints> const &sgrids,
                                 Zip zip)
{
#if !defined(USE_ZLIB)
    if (zip==ZIPPED)
    {
        std::cout << "INFO: zlib not present - vtk file will not be compressed!\n";
        zip = UNZIPPED;
    }
#endif

    // build file name (+rankno) + stepno + vtk extension
    std::stringstream s;
    s << filename;
    s << '_' << std::setw(8) << std::setfill('0') << step << ".pvti";

    // open file
    std::cout << "writing results to " << s.str() << '\n';
    std::ofstream f(s.str().c_str(), std::ios::binary | std::ios::out);
    f << std::scientific;

    // header
    f << "<?xml version=\"1.0\"?>\n";

    f << "<VTKFile type=\"PImageData\" version=\"0.1\" byte_order=\"";
    f << (isCpuLittleEndian ? "LittleEndian" : "BigEndian") << "\" ";
    f << "header_type=\"UInt32\" "; // UInt64 should be better
    if (zip==ZIPPED)
        f << "compressor=\"vtkZLibDataCompressor\" ";
    f << ">\n";

    f << "  <PImageData ";
    f << "WholeExtent=\""
      << grid.np1[0] << ' ' << grid.np2[0]+1 << ' '
      << grid.np1[0] << ' ' << grid.np2[1]+1 << ' '
      << grid.np1[0] << ' ' << grid.np2[2]+1 << "\" ";
    f << "GhostLevel=\"0\" ";
    f << "Origin=\"" << grid.o[0] << ' ' << grid.o[1] << ' ' << grid.o[2] << "\" ";
    f << "Spacing=\"" << grid.dx[0] << ' ' << grid.dx[1] << ' ' << grid.dx[2] << "\">\n";

    // ------------------------------------------------------------------------------------
    f << "      <PCellData>\n";
    // scalar fields
    for (auto it = mygrid.scalars.begin(); it != mygrid.scalars.end(); ++it)
    {
        f << "        <PDataArray type=\"Float32\" ";
        f << " Name=\"" << it->first << "\" />\n";
    }
    // vector fields
    for (auto it = mygrid.vectors.begin(); it != mygrid.vectors.end(); ++it)
    {
        f << "        <PDataArray type=\"Float32\" ";
        f << " Name=\"" << it->first << "\" ";
        f << " NumberOfComponents=\"3\" />\n";
    }
    f << "      </PCellData>\n";

    // ------------------------------------------------------------------------------------
    f << "      <PPointData>\n";
    f << "      </PPointData>\n";

    // ------------------------------------------------------------------------------------

    for (auto it = sgrids.begin(); it != sgrids.end(); ++it)
    {
        f << "    <Piece ";
        f << " Extent=\"";
        f << it->np1[0] << ' ' << it->np2[0]+1 << ' ';
        f << it->np1[1] << ' ' << it->np2[1]+1 << ' ';
        f << it->np1[2] << ' ' << it->np2[2]+1 << "\" ";

        f << "Source=\"";
        std::stringstream s;
        s << filename;
        s << "_r" << it->id;
        s << '_' << std::setw(8) << std::setfill('0') << step << ".vti";
        f << s.str() << "\" />\n";
    }
    // ------------------------------------------------------------------------------------
    f << "  </PImageData>\n";

    f << "</VTKFile>\n";

    f.close();
}
