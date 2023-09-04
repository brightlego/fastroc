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

struct Args {
    char *y_true;
    double *y_score;
    double *roc_auc;
    npy_intp dim0;
    npy_intp dim1;
    npy_intp dim2;
    int offset;
    int integral_precision;
    int thread_count;
};


void *
roc_thread(void *voided_args) {
    struct Args args = * (struct Args *) voided_args;
    npy_intp i0;
    npy_intp i, j, k;
    for (i0 = args.offset; i0 < args.dim0*args.dim2; i0 += args.thread_count) {
        k = i0 % args.dim2;
        i = i0 / args.dim2;
        double auc, tpr, fpr, prev_tpr, prev_fpr, threshold;
        auc = 0;
        prev_tpr = 1;
        prev_fpr = 1;
        int threshold_i;
        int true_count = 0, false_count = 0;
        for (j = 0; j < args.dim1; j++) {
            npy_intp index = i*args.dim1*args.dim2 + j*args.dim2 + k;
            if (args.y_true[index]) {
                true_count++;
            } else {
                false_count++;
            }
        }
        for (threshold_i = 0;
             threshold_i <= args.integral_precision;
             threshold_i++) {
            threshold = (double) threshold_i / args.integral_precision;
            tpr = 0;
            fpr = 0;
            for (j = 0; j < args.dim1; j++) {
                npy_intp index = i*args.dim1*args.dim2 + j*args.dim2 + k;
                if (args.y_score[index] >= threshold) {
                    if (args.y_true[index]) {
                        tpr += 1;
                    } else {
                        fpr += 1;
                    }
                }
            }
            tpr /= true_count;
            fpr /= false_count;
            auc += (fpr - prev_fpr) * 0.5 * (tpr + prev_tpr);
            prev_fpr = fpr;
            prev_tpr = tpr;
        }
        if (auc < 0) {
            auc = -auc;
        }
        args.roc_auc[i0] = auc;
    }
    return NULL;
}

int
calculate_roc(char *y_true, double *y_score, double *roc_auc,
              npy_intp dim0, npy_intp dim1, npy_intp dim2,
              int integral_precision, int thread_count)
{
#ifdef MULTITHREADED
    if (thread_count == 1) {
#else
        if (thread_count != 1) {
            PyErr_WarnEx(PyExc_RuntimeWarning, "Only using single thread as pthread not found.\n", 1);
        }
#endif
        struct Args args = {
                y_true,
                y_score,
                roc_auc,
                dim0,
                dim1,
                dim2,
                0,
                integral_precision,
                thread_count
        };
        roc_thread((void *) &args);
#ifdef MULTITHREADED
    } else {
        pthread_t threads[thread_count];
        struct Args args[thread_count];
        int i;
        for (i = 0; i < thread_count; i++) {
            args[i].y_true = y_true;
            args[i].y_score = y_score;
            args[i].roc_auc = roc_auc;
            args[i].dim0 = dim0;
            args[i].dim1 = dim1;
            args[i].dim2 = dim2;
            args[i].offset = i;
            args[i].integral_precision = integral_precision;
            args[i].thread_count = thread_count;
            pthread_create(threads + i, NULL, roc_thread,
                           (void *) (args + i));
        }
        for (i = 0; i < thread_count; i++) {
            pthread_join(threads[i], NULL);
        }
    }
#endif
    return 0;
}

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


static PyObject *
fastroc_calc_roc_auc(PyObject *self, PyObject *args, PyObject *kwargs) {
    npy_intp dim0, dim1, dim2;

    PyObject *arg1, *arg2;
    PyArrayObject *y_true_arr, *y_score_arr;
    int axis = -1;
    int integral_precision = 50;
    int thread_count = 1;
    char *keywords[] = {"y_true", "y_score", "axis", "integral_precision", "thread_count", NULL};
    PyArg_ParseTupleAndKeywords(args, kwargs, "OO|iii", keywords, &arg1, &arg2, &axis, &integral_precision, &thread_count);

    if (PyArray_Check(arg1)) {
        y_true_arr = PyArray_GETCONTIGUOUS((PyArrayObject *) arg1);
    } else {
        PyErr_Clear();
        PyErr_SetString(PyExc_TypeError, "y_true must be a numpy array");
        return NULL;
    }
    if (PyArray_Check(arg2)) {
        y_score_arr = PyArray_GETCONTIGUOUS((PyArrayObject *) arg2);
    } else {
        PyErr_Clear();
        PyErr_SetString(PyExc_TypeError, "y_score must be a numpy array");
        return NULL;
    }
    if (!PyArray_ISBOOL(y_true_arr)) {
        PyErr_Clear();
        PyErr_SetString(PyExc_TypeError, "y_true must be an array of booleans");
        return NULL;
    }

    if (!(PyArray_ISFLOAT(y_score_arr) && PyArray_ITEMSIZE(y_score_arr) == 8)) {
        PyErr_Clear();
        PyErr_SetString(PyExc_TypeError, "y_score must be an array of floats (float64)");
        return NULL;
    }

    if (PyArray_NDIM(y_score_arr) != PyArray_NDIM(y_true_arr)) {
        PyErr_Clear();
        PyErr_SetString(PyExc_ValueError, "y_score and y_true must have the same shape");
        return NULL;
    }

    int i;
    for (i = 0; i < PyArray_NDIM(y_true_arr); i++) {
        if (PyArray_DIM(y_score_arr, i) != PyArray_DIM(y_true_arr, i)) {
            PyErr_Clear();
            PyErr_SetString(PyExc_ValueError, "y_score and y_true must have the same shape");
            return NULL;
        }
    }

    if (PyArray_NDIM(y_true_arr) < 1) {
        PyErr_Clear();
        PyErr_SetString(PyExc_ValueError, "y_score and y_true must be at least 1 dimensional");
        return NULL;
    }

    if (axis < 0) {
        axis += PyArray_NDIM(y_true_arr);
    }
    if (0 > axis || axis >= PyArray_NDIM(y_true_arr)) {
        PyErr_Clear();
        PyErr_Format(PyExc_ValueError, "axis is out of range for array of size %ld", PyArray_NDIM(y_true_arr));
        return NULL;
    }
    npy_intp dims[PyArray_NDIM(y_true_arr)-1];

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


    PyObject *roc_auc_arr = PyArray_SimpleNew(
            PyArray_NDIM(y_true_arr)-1,
            dims,
            NPY_DOUBLE);
    Py_INCREF(roc_auc_arr);
    char *y_true = (char *) PyArray_DATA(y_true_arr);
    double *y_score = (double *) PyArray_DATA(y_score_arr);
    double *roc_auc = (double *) PyArray_DATA((PyArrayObject *) roc_auc_arr);

    calculate_roc(y_true, y_score, roc_auc, dim0, dim1, dim2, integral_precision, thread_count);

    Py_DECREF(y_true_arr);
    Py_DECREF(y_score_arr);
    return roc_auc_arr;
}

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
PyDoc_STRVAR(fastroc_doc, "Calculate the ROC AUC score very quickly");

PyMethodDef methods[] = {
        {
                "calc_roc_auc",
                fastroc_calc_roc_auc,
                METH_VARARGS | METH_KEYWORDS,
                calc_roc_auc_doc
        }, {NULL}
}
;

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

PyMODINIT_FUNC PyInit_fastroc(void) {
#ifndef MULTITHREADED
    PyErr_WarnEx(PyExc_RuntimeWarning, "Not built with pthreads to multithreading disabled", 1);
#endif
    import_array();
    return PyModule_Create(&fastroc_module);
}


