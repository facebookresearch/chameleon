/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
*
* This source code is licensed under the Chameleon License found in the
* LICENSE file in the root directory of this source tree.
*/
import { Route, Routes } from "react-router-dom";

import { GenerateMixedModal } from "./components/pages/GenerateMixedModal";

import { BasicNavbar } from "./components/ri-components/navbars/BasicNavbar";

// JSON Viewer specific css
import "react18-json-view/src/style.css";

function App() {
  return (
    <div>
      <BasicNavbar
        content={{title: "Chameleon", description: "Model Input/Output Viewer"}}
      />

      <Routes>
        <Route index path="*" element={
            <div className="relative lg:px-12 px-5 flex flex-col">
              <GenerateMixedModal />
            </div>
          }
        />
      </Routes>
    </div>
  );
}

export default App;
