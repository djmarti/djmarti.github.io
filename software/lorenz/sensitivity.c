#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

/* Parameters in Lorenz equations */
struct parameters {
    double sigma;
    double rho;
    double beta;
};

/* Vector field */
int func(double t, const double y[], double f[], void *params)
{
    struct parameters *pars = (struct parameters *) params;
    double sigma = pars->sigma;
    double rho = pars->rho;
    double beta = pars->beta;
    f[0] = sigma * (y[1] - y[0]);
    f[1] = y[0] * (rho - y[2]) - y[1];
    f[2] = y[0] * y[1] - beta * y[2];
    t = 0;                      /* do nothing (avoid warnings) */
    return GSL_SUCCESS;
}

/* Jacobian of the vector field */
int jac(double t, const double y[], double *dfdy, double dfdt[], void *params)
{
    struct parameters *pars = (struct parameters *) params;
    double sigma = (pars->sigma);
    double rho = (pars->rho);
    double beta = (pars->beta);
    gsl_matrix_view dfdy_mat = gsl_matrix_view_array(dfdy, 3, 3);
    gsl_matrix *m = &dfdy_mat.matrix;
    /* Fill the entries in the Jacobian */
    gsl_matrix_set(m, 0, 0, -sigma);
    gsl_matrix_set(m, 0, 1, sigma);
    gsl_matrix_set(m, 0, 2, 0);
    gsl_matrix_set(m, 1, 0, rho);
    gsl_matrix_set(m, 1, 1, -1.0);
    gsl_matrix_set(m, 1, 2, -y[0]);
    gsl_matrix_set(m, 2, 0, y[1]);
    gsl_matrix_set(m, 2, 1, y[0]);
    gsl_matrix_set(m, 2, 2, -beta);
    dfdt[0] = 0.0;
    dfdt[1] = 0.0;
    dfdt[2] = 0.0;
    t = 0;                      /* do nothing (avoid warnings) */
    return GSL_SUCCESS;
}

int main(void)
{
    /* Set up GSL ODE integrator */
    const gsl_odeiv_step_type *T = gsl_odeiv_step_rk8pd;
    gsl_odeiv_step *ode_step = gsl_odeiv_step_alloc(T, 3);
    gsl_odeiv_control *ode_control =
        gsl_odeiv_control_standard_new(1e-10, 0.0, 1, 1);
    gsl_odeiv_evolve *ode_evolve = gsl_odeiv_evolve_alloc(3);

    /* Set up GSL random number generator */
    const gsl_rng_type *R;
    gsl_rng *rnd_instance;
    gsl_rng_env_setup();
    R = gsl_rng_default;
    rnd_instance = gsl_rng_alloc(R);

    /* Number of sample points used in the 'background' trajectory */
    const int bkgd_samples = 20000;

    /* Actual values for the Lorenz parameters */
    struct parameters p;
    p.sigma = 10.;
    p.rho = 27.;
    p.beta = 8. / 3.;

    /* Initialize ODE handler */
    gsl_odeiv_system ode_sys = { func, jac, 3, &p };


    double t = 0.0;             /* time */
    double t1 = 10.0;           /* total simulated time */
    double h = 1e-10;           /* initial timestep (modified */
    double y[3] = { 0.0, 0.1, 10.0 };   /* initial conditions */
    double y0[3];
    for (int i = 0; i < 3; i++) {
        y0[i] = y[i];
    }

    /* ---- Simulate the 'background trajectory'  ---- 
     *      * ----------------------------------------------- 
     *           * The 'background' trajectory is the one we use to
     *                * visualize the strange attractor */

    FILE *af = fopen("attractor.inc", "w");
    fprintf(af, "sphere { <%.6e, %.6e, %.6e>", y[0], y[1], y[2]);
    fprintf(af, ", 0.02 texture { pigment { Black } } }\n");
    for (int i = 1; i <= bkgd_samples; i++) {
        double ri = i / (double) bkgd_samples;  /* normalized time [0,1] */
        double ti = ri * t1;
        while (t < ti) {
            gsl_odeiv_evolve_apply(ode_evolve, ode_control, ode_step, &ode_sys,
                                   &t, ti, &h, y);
        }
        fprintf(af, "sphere { <%.6e, %.6e, %.6e>", y[0], y[1], y[2]);
        fprintf(af,
                ", 0.05 texture { pigment { White } } finish { diffuse .5 phong .75 ambient .2 } }\n");
        fprintf(af, "cylinder {<%.6e, %.6e, %.6e>", y0[0], y0[1], y0[2]);
        fprintf(af, ", <%.6e, %.6e, %.6e>", y[0], y[1], y[2]);
        fprintf(af,
                ", 0.05 texture { pigment { White } } finish { diffuse .5 phong .75 ambient .2 } }\n");

        for (int k = 0; k < 3; k++) {
            y0[k] = y[k];
        }
    }
    fclose(af);

    /* ---- Simulate the ensemble of trajectories ---- */
    /* ----------------------------------------------- */

    const int n_samples = 200;  /* ensemble size */
    int n_shots = 1200;         /* number of snapshots per trajectory */
    double max_time = 12.0;     /* total simulated time */

    /* Initialize 3-rank matrix (axes: n_sample, snapshot_time, xyz coordinates) */
    double ***A;                /* A is a 3-dimensional array */
    A = malloc(n_samples * sizeof(double **));
    /* for each pointer, allocate storage for an array of doubles */
    for (int i = 0; i < n_samples; i++) {
        A[i] = malloc(n_shots * sizeof(double *));
        for (int j = 0; j < n_shots; j++) {
            A[i][j] = malloc(3 * sizeof(double));
        }
    }

    /* Compute trajectories and fill the 3-rank matrix with them */
    double ti, ri;
    for (int i = 0; i != n_samples; i++) {
        t = 0;
        gsl_odeiv_step_reset(ode_step);
        /* Random initial conditions around (0, 0.1, 10) */
        A[i][0][0] = gsl_ran_gaussian(rnd_instance, 0.1);
        A[i][0][1] = 0.1 + gsl_ran_gaussian(rnd_instance, 0.1);
        A[i][0][2] = 10. + gsl_ran_gaussian(rnd_instance, 0.1);
        for (int k = 0; k < 3; k++) {
            y[k] = A[i][0][k];
        }
        /* Snapshots of the ith-trajectory */
        for (int j = 0; j < n_shots; j++) {
            ri = j / (double) n_shots;  /* normalized time [0,1] */
            ti = ri * max_time;
            while (t < ti) {
                gsl_odeiv_evolve_apply(ode_evolve, ode_control, ode_step,
                                       &ode_sys, &t, ti, &h, y);
            }
            for (int k = 0; k < 3; k++) {
                A[i][j][k] = y[k];
            }
        }
    }

    /* output file */
    FILE *of;
    char filename[20];
    for (int j = 0; j < n_shots; j++) {
        snprintf(filename, 20, "cloud_%04d.inc", (j + 1));
        /* printf("Generating %ode_step...\n", filename); */
        of = fopen(filename, "w");
        for (int i = 0; i < n_samples; i++) {
            fprintf(of, "sphere { <%.6e, %.6e, %.6e>", A[i][j][0], A[i][j][1],
                    A[i][j][2]);
            fprintf(of,
                    ", 0.3 texture { crvTex } finish { diffuse .5 phong .75 ambient .3 } }\n");
        }
        fclose(of);
    }

    /* Be clean: free allocated memory */
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_shots; j++) {
            free(A[i][j]);
        }
        free(A[i]);
    }
    free(A);
    gsl_odeiv_evolve_free(ode_evolve);
    gsl_odeiv_control_free(ode_control);
    gsl_odeiv_step_free(ode_step);
    return 0;
}
