#ifndef CTR_CONSTANTS_H
#define CTR_CONSTANTS_H

#include "Eigen/Dense"

namespace CtrLib{
    // Parameters
    constexpr int NB_TUBES = 3;                        // number of tubes (Do not change this parameter, some parts of the code are hard-coded for a 3-tube CTR. Generalization to n-tube CTR will be implemented later.)
    constexpr int NB_INTEGRATION_NODES = 2;               // number of integration nodes per segment (Increasing this value, e.g to 5, yield smoother shaper in the 3d plot, but at the cost of slower computation)

    // Do not change these constants, as they are automatically computed from the parameters.
    constexpr int NB_SEGMENT_MAX = 2 * NB_TUBES;            // max number of segments
    constexpr int NB_STATE_VAR = 15 + 2 * (NB_TUBES - 1);   // number of state variables

    constexpr int NB_Q = 2 * NB_TUBES;                      // number of actuation variables
    constexpr int NB_YU0 = NB_TUBES + 2;                    // number of unknown initial state variables
    constexpr int NB_BC = NB_TUBES + 2;                     // number of variables in boundary condition residuals 

    constexpr int NB_W = 6; // Number of variable in the wrench vector
    constexpr int NB_X = 6; // Number of variable in the pose/twist vector

    constexpr double pi = 3.1415926535897932384626433832795028841971L;

    typedef Eigen::Vector<double,NB_Q>      Vector_q;   // type for actuation
    typedef Eigen::Vector<double,NB_YU0>    Vector_yu0; // type for unknown initial state variables
    typedef Eigen::Vector<double,NB_BC>     Vector_bc;  // type for boundary condition residuals 
    typedef Eigen::Vector<double,NB_W>      Vector_w;   // type for external wrench
    typedef Eigen::Vector<double,NB_X>      Vector_X;   // type for pose/twist

    typedef Eigen::Vector<double,NB_SEGMENT_MAX>                                      Vector_S;
    typedef Eigen::Matrix<double,NB_TUBES, NB_SEGMENT_MAX>                            Matrix_segmented; 
    typedef Eigen::Matrix<double,NB_STATE_VAR, NB_SEGMENT_MAX * NB_INTEGRATION_NODES> Matrix_yTot;

    typedef Eigen::Matrix<double, NB_X, NB_Q>     Matrix_J; // type for robot Jacobian matrix
    typedef Eigen::Matrix<double, NB_X, NB_W>     Matrix_C; // type for Compliance matrix

    typedef Eigen::Matrix<double, NB_X, NB_Q>     Matrix_Eq; // type for Eq matrix
    typedef Eigen::Matrix<double, NB_X, NB_YU0>   Matrix_Eu; // type for Eu matrix
    typedef Eigen::Matrix<double, NB_X, NB_W>     Matrix_Ew; // type for Ew matrix

    typedef Eigen::Matrix<double, NB_BC, NB_Q>    Matrix_Bq; // type for Bq matrix
    typedef Eigen::Matrix<double, NB_BC, NB_YU0>  Matrix_Bu; // type for Bu matrix
    typedef Eigen::Matrix<double, NB_BC, NB_W>    Matrix_Bw; // type for Bw matrix
    
    struct segmentedData{
        Vector_S                        S;    // vector containing the arc length at the end of each segment
        Eigen::Vector<int, NB_TUBES>    iEnd; // vector of the segment index where each tube ends
        Matrix_segmented                KKxy; // matrix of bending stiffness for each tube (row) at each segment (column)
        Matrix_segmented                KKz;  // matrix of torsional stiffness for each tube (row) at each segment (column)
        Matrix_segmented                UUx;  // matrix of precurvature for each tube (row) at each segment (column)
    };

    struct computationOptions{
        bool isExternalLoads;       // true for loaded model, false for unloaded model (WARNING ! Unloaded model is not impleted for the moment)
        bool isComputeJacobian;     // true to compute the robot jacobian matrix 
        bool isComputeCompliance;   // true to compute the compliance matrix 
        int  nbThreads;             // number of threads to use for parallel computing (simply use 1 for single-thread computation)
        };

    struct tubeParameters{
        Eigen::Vector<double, NB_TUBES> Ux;         // Precurvature along x-axis (m^-1) for each tube
        Eigen::Vector<double, NB_TUBES> l;          // Effective length for each tube [m]
        Eigen::Vector<double, NB_TUBES> l_k;        // Precurved length for each tube [m]
        Eigen::Vector<double, NB_TUBES> Kxy;        // Bending stiffness E*I (Pa.m^4) for each tube 
        Eigen::Vector<double, NB_TUBES> Kz;         // Torsional stiffnes G*J (Pa.m^4) for each tube
    };
    // pre-defined options for single-thread computing the loaded model / loaded + Jacobian matrix / loaded + Jacobian and Compliance matrices
    constexpr computationOptions opt_LOAD     = { .isExternalLoads = true, 
                                                  .isComputeJacobian = false, 
                                                  .isComputeCompliance = false, 
                                                  .nbThreads = 1};
    constexpr computationOptions opt_LOAD_J   = { .isExternalLoads = true, 
                                                  .isComputeJacobian = true, 
                                                  .isComputeCompliance = false, 
                                                  .nbThreads = 1};
    constexpr computationOptions opt_LOAD_J_C = { .isExternalLoads = true, 
                                                  .isComputeJacobian = true, 
                                                  .isComputeCompliance = true, 
                                                  .nbThreads = 1};

    // Convert index of segment (in SegmentedData) to column index in yTot matrix
    inline int getYtotIndexFromIend(int iEnd){return iEnd * NB_INTEGRATION_NODES + NB_INTEGRATION_NODES - 1;}

    // get position P from yTot
    inline Eigen::Vector3d getPFromYtot(
        const Matrix_yTot &yTot,
        const segmentedData &segmented){
            
        return yTot(Eigen::seqN(0,3), getYtotIndexFromIend(segmented.iEnd(0)));
    };
    
    // get "flattened" 9-element vector representing R from yTot
    inline Eigen::Vector<double,9> getRvFromYtot(
        const Matrix_yTot &yTot,
        const segmentedData &segmented){
            
        return yTot(Eigen::seqN(3,9), getYtotIndexFromIend(segmented.iEnd(0)));
    };

    // get 3x3 Matrix R from yTot
    inline Eigen::Matrix3d getRFromYtot(
        const Matrix_yTot &yTot,
        const segmentedData &segmented){
            
        return getRvFromYtot(yTot, segmented).reshaped<Eigen::RowMajor>(3,3);
    };
}

#endif //CTR_CONSTANTS_H