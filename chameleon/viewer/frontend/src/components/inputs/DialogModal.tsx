/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
*
* This source code is licensed under the Chameleon License found in the
* LICENSE file in the root directory of this source tree.
*/
import { ReactNode, useRef, useEffect } from "react";
import { Close } from "@carbon/icons-react";

export interface DialogModalProps {
  onHide?: () => void;
  onShow?: () => void;
  visible: boolean;
  children: ReactNode;
}

export function DialogModal({
  onShow,
  onHide,
  visible,
  children,
}: DialogModalProps) {
  const shareRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    if (visible) {
      shareRef.current?.showModal();
      onShow && onShow();
    } else {
      shareRef.current?.close();
      onHide && onHide();
    }
  }, [visible]);

  return (
    <dialog ref={shareRef} className="modal">
      <div className="modal-box relative">
        <div
          className="absolute top-0 right-0 p-4 hover:text-primary cursor-pointer"
          onClick={onHide}
        >
          <Close size={32} />
        </div>
        {children}
      </div>
      <form method="dialog" className="modal-backdrop">
        <button>close</button>
      </form>
    </dialog>
  );
}
