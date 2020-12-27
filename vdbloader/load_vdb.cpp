#include "load_vdb.h"

#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/ValueTransformer.h>
#include <openvdb/tree/LeafManager.h>

#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

class Volume
{
private:
    std::vector<float> _data;
    int                _nx;
    int                _ny;
    int                _nz;
    int                _nxy;
    float              _min_value;
    float              _max_value;

public:
    Volume(int nx, int ny, int nz) : _nx(nx), _ny(ny), _nz(nz), _nxy(nx * ny)
    {
        _data.resize(_nx * _ny * _nz);
    }

    void record_minmax(float min_value, float max_value)
    {
        _min_value = min_value;
        _max_value = max_value;
    }

    size_t       width() { return _nx; }
    size_t       height() { return _ny; }
    size_t       depth() { return _nz; }
    float        min_value() { return _min_value; }
    float        max_value() { return _max_value; }
    const float* data() const { return _data.data(); }

    void set(int i, int j, int k, float v)
    {
        _data[i + j * _nx + k * _nxy] = v;
    }

    void dump(const std::string& filename) const
    {
        std::ofstream ofs(filename.c_str(), std::ios::out | std::ios::binary);
        if (!ofs.is_open())
        {
            std::cout << "cannot write to " << filename << std::endl;
            throw std::runtime_error("write file error");
        }

        std::cout << "writing volume " << _nx << " x " << _ny << " x " << _nz
                  << std::endl;
        ofs.write(reinterpret_cast<const char*>(&_nx), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&_ny), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&_nz), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(_data.data()),
                  sizeof(float) * _data.size());
        ofs.close();
    }
};

Volume* read_voxels(openvdb::FloatGrid::ConstPtr grid)
{
    using GridType = openvdb::FloatGrid;
    using openvdb::Index64;

    typedef GridType::ValueType                  ValueType;
    typedef GridType::TreeType                   TreeType;
    typedef TreeType::ValueConverter<bool>::Type BoolTreeT;
    typedef TreeType::ValueOnCIter               ValueOnCIter;
    typedef TreeType::ValueAllCIter              ValueAllCIter;

    openvdb::CoordBBox box   = grid->evalActiveVoxelBoundingBox();
    openvdb::Coord     start = box.getStart();
    openvdb::Coord     dim   = box.dim();
    auto vol                 = new Volume(dim.x(), dim.y(), dim.z());
    const TreeType& tree     = grid->tree();

    ValueType minValue, maxValue;
    grid->evalMinMax(minValue, maxValue);
    std::cout << "min value: " << minValue << std::endl;
    std::cout << "max value: " << maxValue << std::endl;

    vol->record_minmax(minValue, maxValue);
    openvdb::tree::LeafManager<const TreeType> leafs(tree);
    BoolTreeT::Ptr interiorMask(new BoolTreeT(false));
    interiorMask->topologyUnion(tree);
    interiorMask->voxelizeActiveTiles();
    openvdb::tree::LeafManager<BoolTreeT> maskleafs(*interiorMask);
    const openvdb::tree::LeafManager<BoolTreeT>::RangeType& range = maskleafs.getRange();
    const openvdb::math::Transform& transform = grid->transform();
    openvdb::tree::ValueAccessor<const GridType::TreeType> accessor(grid->tree());

    for (size_t n = range.begin(); n < range.end(); ++n)
    {
        auto    it            = maskleafs.leaf(n).cbeginValueOn();
        Index64 active_voxels = maskleafs.leaf(n).onVoxelCount();
        for (; it; ++it)
        {
            openvdb::Coord ijk = it.getCoord();
            const ValueType& value  = accessor.getValue(ijk);

            // write dense volume
            openvdb::Coord ii = ijk - start;
            vol->set(ii.x(), ii.y(), ii.z(), value);
        }
    }

    return vol;
}

float* load_vdb(char* filename, int& width, int& height, int& depth, float& min_value, float& max_value)
{
    openvdb::initialize();
    openvdb::io::File file(filename);
    file.open();

    openvdb::GridPtrVecPtr grids = file.getGrids();
    if (grids->empty())
    {
        OPENVDB_LOG_WARN(filename << "is empty");
        return nullptr;
    }

    float* data = nullptr;

    for (auto g : *grids)
    {
        // convert the first grid
        auto grid = openvdb::gridPtrCast<openvdb::FloatGrid>(g);
        if (grid)
        {
            auto vol = read_voxels(grid);
            width    = vol->width();
            height   = vol->height();
            depth    = vol->depth();
            min_value = vol->min_value();
            max_value = vol->max_value();
            size_t total_bytes =
                sizeof(float) * vol->width() * vol->height() * vol->depth();
            data = reinterpret_cast<float*>(malloc(total_bytes));
            memcpy(data, vol->data(), total_bytes);
            break;
        }
    }

    return data;
}