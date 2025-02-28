#ifndef CTR_MODEL_H
#define CTR_MODEL_H
#include "Eigen/Dense"
#include "ctrConstants.h"

namespace CtrLib{
    struct parameters; // forward declaration (defined in "loadParameters.h")
    class CtrModel
    {
    public:
        CtrModel(const parameters &p);
        virtual ~CtrModel();

        //template <COMPUTATION_OPTION opt> 
        int Compute(const Vector_q &q, const computationOptions &opt); // Compute end-effector position

        const Vector_q GetOffset(){return offset;};
        const Eigen::Vector<double, NB_TUBES> GetUx(){return Ux;};
        const Eigen::Vector<double, NB_TUBES> GetL(){return l;};
        const Eigen::Vector<double, NB_TUBES> GetL_k(){return l_k;};
        const Eigen::Vector<double, NB_TUBES> GetKxy(){return Kxy;};
        const Eigen::Vector<double, NB_TUBES> GetKz(){return Kz;};
        const Vector_w GetW(){return w;};
        const Matrix_yTot GetYTot(){return yTot;}
        const Vector_yu0 GetYu0(){return yu0;};
        const Eigen::Vector3d GetP(){return P;};
        const Eigen::Matrix3d GetR(){return R;};
        const Matrix_J GetJ(){return J;};
        const Matrix_C GetC(){return C;};

        void SetW(const Vector_w &_w){w = _w;};

        segmentedData segmented;

    protected:
        Eigen::Vector<double, NB_TUBES> Ux;         // Precurvature along x-axis (m^-1) for each tube
        Eigen::Vector<double, NB_TUBES> l;          // Effective length for each tube [m]
        Eigen::Vector<double, NB_TUBES> l_k;        // Precurved length for each tube [m]
        Eigen::Vector<double, NB_TUBES> Kxy;        // Bending stiffness E*I (Pa.m^4) for each tube 
        Eigen::Vector<double, NB_TUBES> Kz;         // Torsional stiffnes G*J (Pa.m^4) for each tube
        Vector_q q;       // Actuation variables [beta_i, alpha_i] with beta the translations (m) and alpha the rotations (rad) actuation
        Vector_bc b;      // Boundary condition errors
        Vector_w w;                         // External wrench applied on the end-effector
        Matrix_yTot yTot; // Matrix containing state variables along arc-length (at each integration node, as defined in "ctrConstants.h")
        Vector_yu0 yu0;   // Guess for unknown initial state variables
        Eigen::Vector3d P;                             // Position of the end-effector
        Eigen::Matrix3d R;                             // Orientation of the end-effector as a rotation matrix

        Matrix_J J;    // Robot Jacobian
        Matrix_C C;    // Robot Compliance

        Vector_q offset; // Translational and rotational offsets (between the "actuator zero position" and the "zero position" defined in the model) [m ; m; m; rad; rad; rad]
    private:
        

    };
}
#endif // CTR_MODEL_H
