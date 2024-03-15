//==============================================================================
//
//  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QNN_HTP_MEMORY_INFRASTRUCTURE_2_H
#define QNN_HTP_MEMORY_INFRASTRUCTURE_2_H

#include "QnnCommon.h"

/**
 *  @file
 *  @brief QNN HTP Memory Infrastructure component API.
 */

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// VTCM
//=============================================================================

// clang-format off

/**
 * @brief Raw memory address that exists ONLY on the QURT
 * side.
 */
typedef uint32_t QnnHtpMem_QurtAddress_t;

// clang-format off

/**
 * @brief QNN Memory Type
 */
typedef enum {
  QNN_HTP_MEM_QURT = 0,
  QNN_HTP_MEM_UNDEFINED = 0x7FFFFFFF
} QnnHtpMem_Type_t;

// clang-format off

/**
 * @brief descriptor used for the QNN API
 */
typedef struct {
  QnnHtpMem_Type_t type;
  uint64_t size;

  union {
    QnnHtpMem_QurtAddress_t qurtAddress;
  };
} QnnMemHtp_Descriptor_t;

// clang-format on
#ifdef __cplusplus
}  // extern "C"
#endif

#endif
