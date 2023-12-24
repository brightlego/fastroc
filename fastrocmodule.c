#define PY_SSIZE_T_CLEAN

#ifdef __has_include
    #if __has_include(<Python.h>)
            #include <Python.h>
    #else
        #include <python3.8/Python.h>
    #endif
#else
    #include <python3.8/Python.h>
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#ifdef __has_include
    #if __has_include(<numpy/arrayobject.h>)
        #include <numpy/arrayobject.h>
    #else
        #include </home/john/.local/lib/python3.8/site-packages/numpy/core\
/include/numpy/arrayobject.h>
    #endif
#else
    #include </home/john/.local/lib/python3.8/site-packages/numpy/core/include\
/numpy/arrayobject.h>
#endif

#ifdef __has_include
    #if __has_include(<pthread.h>)
        #include <pthread.h>
        #define MULTITHREADED
    #endif
#else
    #warning "pthread not found compiling singlethreaded"
#endif

// The arguments to roc_threaded.
// They have to be in a single struct so that pthread can use it
struct Args {
    char *y_true;           // Whether the event happened or not. 0 for does not happen, otherwise happened
    double *y_score;        // The score the program gave for it happening. The lower, the more likely. Between 0 and 1.
    double *roc_auc;        // The array to output the scores to.
    npy_intp dim0;          // The size of the first dimension of the array
    npy_intp dim1;          // The size of the middle dimension of the array (the one to compute the scores over)
    npy_intp dim2;          // The size of the final dimension of the array
    int offset;             // The offset for the index to start at
    int integral_precision; // How many steps to use with the trapezium rule
    int thread_count;       // The number of threads to use
};

// Indexing is done C style (i.e. (0,0,0) is close to (0,0,1) in memory but far from (0,1,0) and very far from (1,0,0)
#define get_index(i, j, k) (((i)*args.dim1 + (j))*args.dim2 + (k)) // gets the index in y_score or y_true

// A single thread of the program
// The Receiver Operating Characteristic (ROC) Area Under Curve (AUC):
/*
 ROC AUC is the area under the ROC curve.
 Let TPC(t) be number of potential events with scores less than t that happened
 Let FPC(t) be number of potential events with scores less that t that did not happen

 The ROC curve is defined parametrically to be:
 x = FPC(t)/FPC(0) = FPR(t)
 y = TPC(t)/TPC(0) = TPR(t)

 for 0 <= t <= 1 (in this code)

 The AUC is the area under this code. This is calculated as Integral of |TPR(t)| with respect to FPR(t).
 In practice, this is done using the trapezium rule:
 the threshold t is sampled at every 1/precision from 0 to 1

 Then the area of the trapezium: (FPR(t), 0); (FPR(t), TPR(t)); (FPR(prev t), TPR(prev t)); (FPR(prev t), 0)
 is added to the total area.

 The ROC AUC score is the value of that area when t=1

 Multithreading is done by each thread calculating every thread_count'th ROC AUC, each starting at a different offset.
 */
void *
roc_thread(void *voided_args)
{
    struct Args args = * (struct Args *) voided_args; // Re-interpret the args as a pointer to an Args structure and dereference it
    npy_intp combined_ik; // Iterating through values of i and k combined to make taking every thread_count'th easier
    npy_intp i, j, k; // The indices of the array (i is dim1, j is dim2, k is dim3)
    for (combined_ik = args.offset; combined_ik < args.dim0 * args.dim2; combined_ik += args.thread_count) {
        double auc;         // The current Area Under Curve (AUC)
        double tpr;         // The current index's True Positive Rate (TPR)
        double fpr;         // The current index's False Positive Rate (FPR)
        double prev_tpr;    // The previous index's True Positive Rate (TPR)
        double prev_fpr;    // The previous index's False Positive Rate (FPR)
        double threshold;   // The current threshold to call an event happening

        // Split combined_ik into i, k using divmod (i.e. i is the dividend, k is the remainder)
        i = combined_ik / args.dim2;
        k = combined_ik % args.dim2;

        auc = 0; // The current AUC
        prev_tpr = 1; // The previous TPR (t=0 should mark every event has predicted to happen)
        prev_fpr = 1; // The previous FPR

        // Calculate TPC(0) and FPC(0) for use in calculating TPR and FPR respectively
        int true_count = 0, false_count = 0;
        for (j = 0; j < args.dim1; j++) {
            npy_intp index = get_index(i, j, k);

            if (args.y_true[index]) {
                true_count++;
            } else {
                false_count++;
            }
        }
        int threshold_i; // The i'th threshold
        for (threshold_i = 0;
             threshold_i <= args.integral_precision;
             threshold_i++) {
            threshold = (double) threshold_i / args.integral_precision;

            // Calculate TPC(threshold), FPC(threshold)
            tpr = 0;
            fpr = 0;
            for (j = 0; j < args.dim1; j++) {
                npy_intp index = get_index(i, j, k);
                if (args.y_score[index] >= threshold) {
                    if (args.y_true[index]) {
                        tpr += 1;
                    } else {
                        fpr += 1;
                    }
                }
            }
            // Divide by TPC(0), FPC(0) to get TPR(threshold), FPR(threshold)
            tpr /= true_count;
            fpr /= false_count;

            // Get the area of the current trapezium
            double curr_trapezium_area = (fpr - prev_fpr) * 0.5 * (tpr + prev_tpr);

            // abs that and add it onto the auc
            auc += curr_trapezium_area < 0 ? -curr_trapezium_area : curr_trapezium_area;

            // Set the previous fpr, tpr to the values of the current fpr, tpr respectively
            prev_fpr = fpr;
            prev_tpr = tpr;
        }

        // set roc_auc[i, k] to the current auc
        args.roc_auc[combined_ik] = auc;
    }
    // Return NULL ((void *) 0) as pthread insists that this function must return something
    return NULL;
}

// Calculates the ROC AUC score for the given arrays. Takes roughly the same arguments as roc_thread
int // returns the status code of the program (at the moment, always 0)
calculate_roc(char *y_true, double *y_score, double *roc_auc,
              npy_intp dim0, npy_intp dim1, npy_intp dim2,
              int integral_precision, int thread_count)
{
#ifdef MULTITHREADED
    // If the program is not compiled with multithreading or only a single thread is used, don't create a new thread
    // just calculate the ROC AUC in the current thread
    if (thread_count == 1) {
#else
        if (thread_count != 1) {
            // Warn the user if they try to use multithreading but there is only a single thread
            PyErr_WarnEx(PyExc_RuntimeWarning, "Only using single thread as pthread not found.\n", 1);
        }
#endif
        // Put the arguments into a struct and call roc_thread with that
        struct Args args = { y_true, y_score, roc_auc, dim0, dim1, dim2, 0, integral_precision, 1 };
        roc_thread((void *) &args);
#ifdef MULTITHREADED
    } else {
        pthread_t threads[thread_count]; // An array to store all the threads
        struct Args args[thread_count]; // An array to store all the arguments to those threads
        int i; // The current index of the thread
        for (i = 0; i < thread_count; i++) {
            // Set the i'th argument.
            args[i] = (struct Args) { y_true, y_score, roc_auc, dim0, dim1, dim2, i, integral_precision,
                                      thread_count };

            pthread_create(threads + i, // &threads[i]
                           NULL, // This is not needed
                           roc_thread, // The function to start the thread from
                           (void *) (args + i)); // &args[i]
        }
        // Now wait for all the threads to finish (and discard the return value)
        for (i = 0; i < thread_count; i++) {
            pthread_join(threads[i], NULL);
        }
    }
#endif
    return 0; // Return 0 (successfully completed)
}

/*

// Code for testing in C to allow debugger usage

int test(int print_results, double y_score_accuracy, int integral_precision) {
    srand(1);
    npy_intp dim0 = 30;
    npy_intp dim1 = 40;
    npy_intp dim2 = 36;
    char *y_true = malloc(sizeof(char) * dim0*dim1*dim2);
    double *y_score = malloc(sizeof(double) * dim0*dim1*dim2);
    double *roc_auc = malloc(sizeof(double) * dim0*dim2);
    int i, j, k;
    printf("Got here!\n");
    for (i = 0; i < dim0; i++) {
        for (j = 0; j < dim1; j++) {
            for (k = 0; k < dim2; k++) {
                double p = ((double) rand()) / RAND_MAX;
                y_true[i*dim1*dim2 + j*dim2 + k] = p < 0.2 ? 1 : 0;
                y_score[i*dim1*dim2 + j*dim2 + k] =
                        (p < 0.2 ? y_score_accuracy / 10 : 0)
                        + ((double) rand()) / RAND_MAX / 10;
            }
        }
    }
    struct timespec start, stop;
    clock_gettime(CLOCK_REALTIME, &start);
    calculate_roc(y_true, y_score, roc_auc, dim0, dim1, dim2,
                  integral_precision, 1);
    clock_gettime(CLOCK_REALTIME, &stop);
    long int nsec = (stop.tv_sec - start.tv_sec)*1000000000
                  + (stop.tv_nsec - start.tv_nsec);
    printf("Time taken: %fms\n", nsec/1000000.0);
    if (print_results) {
        for (i = 0; i < dim0*dim2; i++) {
            printf("%f ", roc_auc[i]);
        }
        printf("\n");
    }
    free(y_true);
    free(y_score);
    free(roc_auc);
    return 0;
}

int main() {
    test(1, 0.5, 25);
    test(1, 0.5, 50);
    test(1, 0.5, 100);

    return 0;
}
*/

// the public function that can be accessed in python
static PyObject * // The numpy array that is returned
fastroc_calc_roc_auc(PyObject *args,     // The arguments to the function
                     PyObject *kwargs) { // The keyword arguments to that funciton
    npy_intp dim0, dim1, dim2; // The (adjusted) dimensions of the array

    PyObject *unvalidated_y_true_arr, *unvalidated_y_score_array; // The first two arguments
    PyArrayObject *y_true_arr, *y_score_arr; // The respective arrays
    int axis = -1; // The axis of -1 indicates to calculate over the final axis
    int integral_precision = 50; // A precision of 50 indicates to take 50 steps. This typically gets it to 2-3 dp.
    int thread_count = 1; // The number of threads to run

    // A NULL terminated array of keyword arguments
    char *keywords[] = {"y_true", "y_score", "axis", "integral_precision", "thread_count", NULL};
    PyArg_ParseTupleAndKeywords(args,     // The non-keyword arguments
                                kwargs,   // The keyword arguments
                                "OO|iii", // 2 mandatory objects then 3 optional integers
                                keywords, // The keywords for the arguments
                                &unvalidated_y_true_arr,
                                &unvalidated_y_score_array,
                                &axis,
                                &integral_precision,
                                &thread_count);


    // Make sure y_true and y_score are, indeed, numpy arrays
    if (PyArray_Check(unvalidated_y_true_arr)) {
        y_true_arr = PyArray_GETCONTIGUOUS((PyArrayObject *) unvalidated_y_true_arr);
    } else {
        PyErr_Clear();
        PyErr_SetString(PyExc_TypeError, "y_true must be a numpy array");
        return NULL;
    }
    if (PyArray_Check(unvalidated_y_score_array)) {
        y_score_arr = PyArray_GETCONTIGUOUS((PyArrayObject *) unvalidated_y_score_array);
    } else {
        PyErr_Clear();
        PyErr_SetString(PyExc_TypeError, "y_score must be a numpy array");
        return NULL;
    }

    // Make sure y_true is an array of bools
    if (!PyArray_ISBOOL(y_true_arr)) {
        PyErr_Clear();
        PyErr_SetString(PyExc_TypeError, "y_true must be an array of booleans");
        return NULL;
    }

    // Make sure y_score is an array of floats
    if (!(PyArray_ISFLOAT(y_score_arr) && PyArray_ITEMSIZE(y_score_arr) == 8)) {
        PyErr_Clear();
        PyErr_SetString(PyExc_TypeError, "y_score must be an array of floats (float64)");
        return NULL;
    }

    // Make sure y_true and y_score have the same number of dimensions
    if (PyArray_NDIM(y_score_arr) != PyArray_NDIM(y_true_arr)) {
        PyErr_Clear();
        PyErr_SetString(PyExc_ValueError, "y_score and y_true must have the same shape");
        return NULL;
    }

    // Make sure y_true and y_score have the same shape
    int i;
    for (i = 0; i < PyArray_NDIM(y_true_arr); i++) {
        if (PyArray_DIM(y_score_arr, i) != PyArray_DIM(y_true_arr, i)) {
            PyErr_Clear();
            PyErr_SetString(PyExc_ValueError, "y_score and y_true must have the same shape");
            return NULL;
        }
    }

    // Make sure that there is an axis to potentially calculate ROC AUC over
    if (PyArray_NDIM(y_true_arr) < 1) {
        PyErr_Clear();
        PyErr_SetString(PyExc_ValueError, "y_score and y_true must be at least 1 dimensional");
        return NULL;
    }

    // Allow negative indexing for the axis to wrap around to the end of the tuple
    if (axis < 0) {
        axis += PyArray_NDIM(y_true_arr);
    }

    // Check that the axis is indeed a valid dimension
    if (0 > axis || axis >= PyArray_NDIM(y_true_arr)) {
        PyErr_Clear();
        PyErr_Format(PyExc_ValueError, "axis %d is out of range for array of size %ld", axis, PyArray_NDIM(y_true_arr));
        return NULL;
    }
    npy_intp dims[PyArray_NDIM(y_true_arr)-1]; // The dimensions of the output array

    // Interpret the data as a 3-dimensional array by merging all the dimensions preceding and succeeding the axis to
    // dim0 and dim2 respectively and computing ROC AUC over dim2. If axis = 0 or -1, then dim0 or dim2 = 1
    // respectively. This allows for any position of axis while the C code only having to cover the 3-dimensional case.
    dim0 = 1; dim2 = 1;
    for (i = 0; i < axis; i++) {
        dim0 *= PyArray_DIM(y_true_arr, i);
        dims[i] = PyArray_DIM(y_true_arr, i);
    }
    dim1 = PyArray_DIM(y_true_arr, axis);
    for (i = axis+1; i < PyArray_NDIM(y_true_arr); i++) {
        dim2 *= PyArray_DIM(y_true_arr, i);
        dims[i-1] = PyArray_DIM(y_true_arr, i);
    }

    // Create a new array for the output
    PyObject *roc_auc_arr = PyArray_SimpleNew(
            PyArray_NDIM(y_true_arr)-1,
            dims,
            NPY_DOUBLE);
    // Increment the reference count for the new array
    Py_INCREF(roc_auc_arr);

    // Get the data arrays for y_true, y_score and roc_auc
    char *y_true = (char *) PyArray_DATA(y_true_arr);
    double *y_score = (double *) PyArray_DATA(y_score_arr);
    double *roc_auc = (double *) PyArray_DATA((PyArrayObject *) roc_auc_arr);

    // Calculate roc_auc
    calculate_roc(y_true, y_score, roc_auc, dim0, dim1, dim2, integral_precision, thread_count);

    // Finish using y_true_arr and y_score_arr
    Py_DECREF(y_true_arr);
    Py_DECREF(y_score_arr);
    return roc_auc_arr;
}

// Generate the docstrings for the code and the module
PyDoc_STRVAR(calc_roc_auc_doc,
 "calc_roc_auc(y_true: numpy.ndarray, y_score: np.ndarray, "
 "axis: int = -1, integral_precision: int = 50, "
 "thread_count: int = 1) -> numpy.ndarray\n\n"
 "Calculate the ROC AUC score from y_true (the event array) and \n"
 "y_score (the score array) along a specified axis.\n"
 "y_true should be an array of booleans and y_score should be an\n"
 "array of float64.\n\n"
 "integral_precision is the number of samples taken to compute the \n"
 "AUC.\n\n"
 "thread_count is the number of extra threads used in calculation.\n"
 "If thread_count is 1 or pthreads is not found then no extra \n"
 "threads are created.\n\n"
             );
PyDoc_STRVAR(fastroc_doc, "Calculate the ROC AUC score very quickly using C");

// Add fastroc_calc_roc_auc to the methods
PyMethodDef methods[] = {
        {
                "calc_roc_auc",
                fastroc_calc_roc_auc,
                METH_VARARGS | METH_KEYWORDS,
                calc_roc_auc_doc
        }, {NULL}
}
;

// Create the module
PyModuleDef fastroc_module = {
        PyModuleDef_HEAD_INIT,
        "fastroc",
        fastroc_doc,
        -1,
        methods,
        NULL,
        NULL,
        NULL,
        NULL
};

// Run when the module is initialised
PyMODINIT_FUNC PyInit_fastroc(void) {
#ifndef MULTITHREADED
    // Warn the user if there is no multithreading
    PyErr_WarnEx(PyExc_RuntimeWarning, "Not built with pthreads to multithreading disabled", 1);
#endif
    import_array(); // Make numpy arrays work
    return PyModule_Create(&fastroc_module);
}


