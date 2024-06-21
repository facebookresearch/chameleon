/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
*
* This source code is licensed under the Chameleon License found in the
* LICENSE file in the root directory of this source tree.
*/
import { Route, Routes } from "react-router-dom";

import { GenerateMixedModal } from "./components/pages/GenerateMixedModal";

import { BasicNavbar, NavContent } from "./components/ri-components/navbars/BasicNavbar";

// JSON Viewer specific css
import "react18-json-view/src/style.css";

function App() {
  const navContent: NavContent = {
    title: "Chameleon",
    description: "Model Input/Output Viewer",
    githubLink: "https://github.com/facebookresearch/chameleon",
    showHomeLink: true,
    navItems: [
      {
        id: "paper-item",
        url: "https://arxiv.org/abs/2405.09818",
        title: "Discover how it works",
        showArrowIcon: true,
      },
    ],
  };

  return (
    <div>
      <BasicNavbar
        content={navContent}
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
