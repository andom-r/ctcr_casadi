#ifndef CTR_MODEL_H
#define CTR_MODEL_H
#include "Eigen/Dense"
#include "ctrConstants.h"
#include "ComputeIvpJacobianMatrices.h"

namespace CtrLib{
    struct parameters; // forward declaration (defined in "loadParameters.h")

    class CtrModel
    {
    public:
        CtrModel(const parameters &p);
        virtual ~CtrModel();

        //template <COMPUTATION_OPTION opt> 
        int Compute(const Vector_q &q, const computationOptions &opt); // Compute end-effector position

        int ComputeIVP(
            const Vector_q &argQ,
            const Vector_yu0 &_yu0, 
            const computationOptions &opt, 
            SingleIVPOut &out
        );

        const Vector_q GetOffset(){return offset;};
        const tubeParameters GetTubeParameters(){return tubes;};
        const Vector_w GetW(){return w;};
        const Matrix_yTot GetYTot(){return yTot;}
        const Vector_yu0 GetYu0(){return yu0;};
        const Eigen::Vector3d GetP(){return P;};
        const Eigen::Matrix3d GetR(){return R;};
        const Matrix_J GetJ(){return J;};
        const Matrix_C GetC(){return C;};
        const Vector_q GetQ(){return q;};

        void SetW(const Vector_w &_w){w = _w;};

        segmentedData segmented;

    protected:
        tubeParameters tubes;
        Vector_q q;        // Actuation variables [beta_i, alpha_i] with beta the translations (m) and alpha the rotations (rad) actuation
        Vector_bc b;       // Boundary condition errors
        Vector_w w;        // External wrench applied on the end-effector (expressed in the base frame)
        Matrix_yTot yTot;  // Matrix containing state variables along arc-length (at each integration node, as defined in "ctrConstants.h")
        Vector_yu0 yu0;    // Guess for unknown initial state variables
        Eigen::Vector3d P; // Position of the end-effector
        Eigen::Matrix3d R; // Orientation of the end-effector as a rotation matrix

        Matrix_J J;        // Robot Jacobian
        Matrix_C C;        // Robot Compliance

        Vector_q offset;   // Translational and rotational offsets (between the "actuator zero position" and the "zero position" defined in the model) [m ; m; m; rad; rad; rad]
    private:
        

    };
}
#endif // CTR_MODEL_H
